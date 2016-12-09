# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Tutorial sample #2: Run simple mission using raw XML

import MalmoPython
import os
import sys
import time
import json
import random
sys.path.append("functions/.")
from DeepAgent import DeepAgent
from deep_q import  DeepLearner
import tensorflow as tf

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately

# Create default Malmo objects:
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print 'ERROR:',e
    print agent_host.getUsage()
    exit(1)
if agent_host.receivedArgument("help"):
    print agent_host.getUsage()
    exit(0)
    
agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)
agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)

# -- set up the mission -- #
mission_file = './mission_setup.xml'
with open(mission_file, 'r') as f:
    print "Loading mission from %s" % mission_file
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
my_mission_record = MalmoPython.MissionRecordSpec()

deep_learner = DeepLearner()
num_repeats = 5000
kills = 0
t = 0

for i in xrange(num_repeats):
    
    prev_kills = kills
    t = deep_learner.t
    first = True
    deep_learner.agent = DeepAgent()
    deep_learner.agent.kills = kills
    print
    print 'Repeat %d of %d' % ( i+1, num_repeats )
# Attempt to start a mission:
    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print "Error starting mission:",e
                exit(1)
            else:
                time.sleep(2)

    # Loop until mission starts:
    print "Waiting for the mission to start ",
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        sys.stdout.write(".")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print "Error:",error.text
    print "Mission running ",
    #Loop until mission ends:
    
    for i in xrange(-3, -1):
        for j in xrange(-3,-1):
            agent_host.sendCommand("chat /summon Zombie " + str(i) + " 207 " +  str(j) +  ' {Equipment:[{},{},{},{},{id:minecraft:stone_button}], HealF:10.0f}')
    agent_host.sendCommand("chat /summon Zombie -3 207 -1 {Equipment:[{},{},{},{},{id:minecraft:stone_button}], HealF:10.0f}")
    agent_host.sendCommand("chat /gamerule naturalRegeneration false")
    agent_host.sendCommand("chat /difficulty 1")
    
    
    
    while world_state.is_mission_running:
        agent_host.sendCommand("attack 1")
        time.sleep(0.03)
        if len(world_state.observations) > 0 and len(world_state.video_frames) > 0:
            if first == True:   
                ob = json.loads(world_state.observations[-1].text)                
                frame = world_state.video_frames[0]                
                action = deep_learner.initNetwork(frame, ob, False)
                agent_host.sendCommand(deep_learner.agent.actions[action])
                first = False
            else:
                ob = json.loads(world_state.observations[-1].text)
                frame = world_state.video_frames[0]
                prev_action = action
                action = deep_learner.trainNetwork(frame, ob, False)
                #print action
                agent_host.sendCommand(deep_learner.agent.antiActions[prev_action])
                agent_host.sendCommand(deep_learner.agent.actions[action])
            
            if "MobsKilled" in ob and ob[u'MobsKilled'] > kills:
                agent_host.sendCommand("chat /summon Zombie -1 207 -3 {Equipment:[{},{},{},{},{id:minecraft:stone_button}], HealF:10.0f}")
                kills = ob[u'MobsKilled']
            
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print "Error:",error.text
    
    deep_learner.trainNetwork(frame, ob, True)
    print "We scored " + str(deep_learner.agent.cum_reward)
    print "We Killed " + str(kills - prev_kills)
    print "We survived for " + str(deep_learner.t - t)

    file = open("rewards.txt", "a")
    file.write(str(deep_learner.agent.cum_reward) + " " + str(kills-prev_kills) + " " + str(deep_learner.t - t) + "\n")
    file.close()
        
    
print
print "Mission ended"
# Mission has ended.