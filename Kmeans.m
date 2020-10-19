rng ('default');
X = readtable('C:\Users\17196\Documents\20 Fall\HPC\Project\nba_stats\teams_Season2014_15_adv.csv');
%X = [randn(100,2)*0.75+ones(100,2);
 %   randn(100,2)*0.5-ones(100,2);
  %  randn(100,2)*0.75];
tempX= table2array(X(:,{'PTS' 'FG_'}));
[idx,C] = kmeans(tempX,3);
figure
gscatter(tempX(:,1),tempX(:,2),idx,'bgc')
hold on
plot(C(:,1),C(:,2),'kx')
legend('Cluster 1','Cluster 2','Cluster 3', 'Cluster Centroid')
title('K-means clustering of NBA 2015 Season');
xlabel('PTS');
ylabel('FG%');

csvwrite('cluster_result.csv',idx);