rng ('default');
disp('Note: if the stat you want has "%" or "/" use a "_" instead');
%input X axis value
prompt = 'X axis stat: ';
Xaxis = input(prompt,'s');
%input Y axis value
prompt = 'Y axis stat: ';
Yaxis = input(prompt,'s');
%input year
Year2 = input('Year(ie: 2014-2015->2015): ');
Year1 = Year2-1;
subYear2 = mod(Year2,100);
X = readtable(strcat('C:\Users\17196\Documents\20 Fall\HPC\Project\nba_stats\teams_Season',num2str(Year1),'_',num2str(subYear2),'_adv.csv'));

tempX= table2array(X(:,{Xaxis Yaxis}));
[idx,C] = kmeans(tempX,3);
figure
gscatter(tempX(:,1),tempX(:,2),idx,'bgc')
hold on
plot(C(:,1),C(:,2),'kx')
legend('Cluster 1','Cluster 2','Cluster 3', 'Cluster Centroid')
title(strcat('K-means clustering of NBA ',num2str(Year2),' Season'));
xlabel(Xaxis);
ylabel(Yaxis);

csvwrite(strcat('cluster_result',num2str(Year1),'_',num2str(subYear2),'_',Xaxis,'_',Yaxis,'.csv'),idx);
