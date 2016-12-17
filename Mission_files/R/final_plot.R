library(ggplot2)
library(zoo)

data = read.csv("final_reward.csv", header=T)
data$index = 1:nrow(data)
#data$time[data$time > 800] = 400

temp.zoo<-zoo(data$complex_score,data$index)
m.av<-rollmean(temp.zoo, 40,fill = list(NA, NULL, NA))
data$complex_score_moving = coredata(m.av)

temp.zoo<-zoo(data$complex_kills,data$index)
m.av<-rollmean(temp.zoo, 40,fill = list(NA, NULL, NA))
data$complex_kills_moving = coredata(m.av)

temp.zoo<-zoo(data$complex_time,data$index)
m.av<-rollmean(temp.zoo, 40,fill = list(NA, NULL, NA))
data$complex_time_moving = coredata(m.av)

temp.zoo<-zoo(data$easy_score,data$index)
m.av<-rollmean(temp.zoo, 40,fill = list(NA, NULL, NA))
data$easy_score_moving = coredata(m.av)

temp.zoo<-zoo(data$easy_kills,data$index)
m.av<-rollmean(temp.zoo, 40,fill = list(NA, NULL, NA))
data$easy_kills_moving = coredata(m.av)

temp.zoo<-zoo(data$easy_time,data$index)
m.av<-rollmean(temp.zoo, 40,fill = list(NA, NULL, NA))
data$easy_time_moving = coredata(m.av)

temp.zoo<-zoo(data$hard_score,data$index)
m.av<-rollmean(temp.zoo, 40,fill = list(NA, NULL, NA))
data$hard_score_moving = coredata(m.av)

temp.zoo<-zoo(data$hard_kills,data$index)
m.av<-rollmean(temp.zoo, 40,fill = list(NA, NULL, NA))
data$hard_kills_moving = coredata(m.av)

temp.zoo<-zoo(data$hard_time,data$index)
m.av<-rollmean(temp.zoo, 40,fill = list(NA, NULL, NA))
data$hard_time_moving = coredata(m.av)

data$complex_easy_score[data$complex_easy_score > 800] = 600
data$hard_easy_score[data$hard_easy_score > 800] = 600

#data[data>1000]= 600
png('5vs9.png')
ggplot(data=data, aes(x=complex_easy_index, y=complex_easy_score, color = complex_easy, shape=complex_easy)) + 
    geom_point(size=0.3, shape = 1)+
    scale_colour_manual(values = c("#F8766D","#00BFC4","#00BFC4" ,"#F8766D"))+
    geom_line(aes(y=complex_score_moving,colour="9 Action"), size=1.25)+ 
    geom_line(aes(y=easy_score_moving,colour="5 Action"), size=1.25)+ 
    ylab("Cumulative Reward") + xlab("Episode") + 
    guides(shape=FALSE)+
    xlim(0,1300)+
    theme(legend.title=element_blank())
dev.off()

png('easyVShard.png')
ggplot(data=data[1:2422,], aes(x=hard_easy_index, y=hard_easy_score, color = hard_easy, shape=hard_easy)) + 
    geom_point(size=0.3, shape = 1)+
    scale_colour_manual(values = c("#F8766D","#00BFC4","#00BFC4","#F8766D"))+
    geom_line(aes(y=hard_score_moving,x = index, colour="Hard"), size=1.25)+ 
    geom_line(aes(y=easy_score_moving, x= index, colour="Easy"), size=1.25)+
    ylab("Cumulative Reward") + xlab("Episode") + 
    guides(shape=FALSE)+
    xlim(0,1100)+
    theme(legend.title=element_blank())
dev.off()


png('qvalue.png')
qvalue = read.csv("final_qvalue.csv", header=T)
qvalue$index = 1:nrow(qvalue)
ggplot(data = qvalue, aes(x=index))+
    geom_line(aes(y=easy_qvalue, colour= "5 Action, Easy"),size = 0.7)+
    geom_line(aes(y=hard_qvalue, colour= "5 Action, Hard"),size = 0.7)+
    geom_line(aes(y=complex_qvalue, colour= "9 Action, Easy"),size = 0.7)+
    ylab("Average Action Value (Q)") + xlab("Frames (1000's)")+
    theme(legend.title=element_blank())
dev.off()



