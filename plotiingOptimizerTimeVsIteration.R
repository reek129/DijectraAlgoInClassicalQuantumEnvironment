library(readxl)
#change filepath according to Excel path
op <- read_xlsx("C:\\Users\\Reek's\\Desktop\\US\\Fall 2019\\DataBase Design\\final\\Optimizer Results.xlsx")
op <- data.frame(op)
unique(op['optimizer'])
op <- op[complete.cases(op['optimizer']=="SPSA"),]
spsa <- data.frame(op[op['optimizer']=='SPSA',])
cobyala <- data.frame(op[op['optimizer']=='Cobyla',])

plot(x=spsa$Iteration,y=spsa$Time,col='blue',type = 'l',xlab ='Iteration',ylab = 'Time' )
lines(x=cobyala$Iteration,y=cobyala$Time,col='red',type='l')

legend("topleft", legend=c("SPSA", "COBYLA"),
       col=c( "blue","red"),lty = 1,cex = 0.7)

