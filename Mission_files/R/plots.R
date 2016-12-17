library(ggplot2)
library(zoo)

data = read.table("rewards.txt", header=T)
data$index = 1:nrow(data)
data$time[data$time > 800] = 400

temp.zoo<-zoo(data$score,data$index)
m.av<-rollmean(temp.zoo, 40,fill = list(NA, NULL, NA))
data$score_moving = coredata(m.av)

temp.zoo<-zoo(data$kills,data$index)
m.av<-rollmean(temp.zoo, 40,fill = list(NA, NULL, NA))
data$kills_moving = coredata(m.av)

temp.zoo<-zoo(data$time,data$index)
m.av<-rollmean(temp.zoo, 40,fill = list(NA, NULL, NA))
data$time_moving = coredata(m.av)

data$score[data$score > 750] = 600
png('reward.png')
ggplot(data=data, aes(x=index, y=score, group=1)) + 
    geom_point(colour="black", size=0.5, shape=21, fill="white")+
    geom_line(aes(y=score_moving,colour="Moving Average"), size=1.25)+ 
    ylab("Total Reward") + xlab("Episode")
dev.off()

png('kills.png')
ggplot(data=data, aes(x=index, y=kills, group=1)) + 
    geom_point(colour="black", size=0.5, shape=21, fill="white")+
    geom_line(aes(y=kills_moving,colour="Moving Average"), size=1.25)+ 
    ylab("Kills") + xlab("Episode") 
dev.off()

png('time.png')
ggplot(data=data, aes(x=index, y=time, group=1)) + 
    geom_point(colour="black", size=0.5, shape=21, fill="white")+
    geom_line(aes(y=time_moving, colour="Moving Average"), size=1.25)+ 
    ylab("Frame Alive") + xlab("Episode") 
dev.off()

#png('Qvalue.png')
qvalue = read.table("qvalue.txt", header=T)
qvalue$index = 1:nrow(qvalue)
ggplot(data = qvalue, aes(x=index, y=qvalue))+
    geom_line(aes(y=qvalue, colour= "Average Action Value"),size = 1)+
    ylab("Average Action Value (Q)") + xlab("Frames (1000's)")+
    ggtitle("Average Action Value Over Frames")
#dev.off()
    
