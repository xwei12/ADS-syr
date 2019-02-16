filename<-file.choose()
dataset<-read.csv(filename,stringsAsFactors = FALSE)
library(dplyr)
library("RColorBrewer")
dataset <- sapply( dataset, as.numeric )
rownames(dataset)<-as.character(heatmap_row)
colnames(dataset)<-c("nb","nb(lowercase = False)","nb(lowercase = False, number to \'@\')","svm","svm(lowercase = False)","svm(lowercase = False, number to \'@\')")
par(mar=c(10,4.1,4.1,2.1))
heatmap(as.matrix(dataset),Rowv=NA,Colv=NA,scale="column",margin=c(25,10),
        col= colorRampPalette(brewer.pal(8, "Blues"))(25))

dat2 <- melt(dataset, id.var = "X1")
ggplot(dat2, aes(Var1, Var2)) +
       geom_tile(aes(fill = value)) + 
       geom_text(aes(label = round(value*100, 4))) +
     scale_fill_gradient(low = "#E8F1FA", high = "#084594") 


cm<-read.csv("confusion_matrix.csv",header=FALSE)
cm <- sapply( cm, as.numeric )
colnames(cm)=1:50
rownames(cm)=1:50
dat3 <- melt(cm, id.var = "X1")
ggplot(dat3, aes(Var1, Var2)) +
  geom_tile(aes(fill = value)) + 
  scale_fill_gradient(low = "#E8F1FA", high = "#084594") + theme_bw() +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

#geom_text(aes(label = round(value*100, 4))) +