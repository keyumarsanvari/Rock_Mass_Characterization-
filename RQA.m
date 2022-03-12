function [Dist,RM,qs]=RQA(X,~)
alpha=0.25;

X=bsxfun(@rdivide,X,sum(X)); %normalisation (importnat when we have multi-variable input)

N = size(X,1);

Dist=zeros(N);

%%%% Norm Matrix %%%%
for i=1:N
    
    x0=i;
    for j=i:N
        y0=j;
        % Calculate the euclidean distance
        distance = norm(X(i,:)-X(j,:));
        % Store the minimum distance between the two points
        Dist(x0,y0) = distance;
        Dist(y0,x0) = distance;        
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Ploting the Distance (Norm) Matrix)%%%%%

    figure('Position',[100 100 550 400]);
    imagesc(Dist);
    colorbar;
    axis image;    
    xlabel('Depth Index','FontSize',10,'FontWeight','bold');
    ylabel('Depth Index','FontSize',10,'FontWeight','bold');
    title('Norm Matrix','FontSize',10,'FontWeight','bold');
    get(gcf,'CurrentAxes');
    set(gca,'YDir','normal')
    set(gca,'LineWidth',2,'FontSize',10,'FontWeight','bold');


%%%%% Define threshold %%%%

threshold=alpha*(mean(Dist(:))+3*std(Dist(:)));

%%%%% Recurrence plot matrix %%%%
RM=zeros(size(Dist,1),size(Dist,2));
for i=1:size(Dist,1)       
    for j=i+1:size(Dist,1)
        if Dist(i,j) <= threshold
            RM(i,j)=1;
            RM(j,i)=1;
        end
    end
end
RM = RM +eye(size(Dist,1));

%%%% Ploting the recurrence plot %%%
    figure('Position',[100 100 550 400]);
    imagesc(RM);
    axis image;  
    xlabel('Depth Index','FontSize',10,'FontWeight','bold');
    ylabel('Depth Index','FontSize',10,'FontWeight','bold');
    title('Recurrence Plot','FontSize',10,'FontWeight','bold');
    get(gcf,'CurrentAxes');
    set(gca,'YDir','normal')
    set(gca,'LineWidth',2,'FontSize',10,'FontWeight','bold');

%%%% Quadrant Scan %%%%
 qs=QS(RM);

%%%% Ploting the quadrant scan %%%%
figure('Position',[100 100 550 400]);
plot(qs)
xlabel('Depth(m)','FontSize',10,'FontWeight','bold'); 
ylabel('quad(dq)','FontSize',10,'FontWeight','bold');
title('Quadrant Scan','FontSize',10,'FontWeight','bold');

%%%%% local peak %%%%%
findpeaks(qs);

%%%%% ML %%%%%
[idx,C] = kmeans(X,k);
% k number of cluster
figure
gscatter(X(:,1),X(:,2),idx,'bgm')
hold on
plot(C(:,1),C(:,2),'kx')
% n number of rocks
legend('ROCK n','Cluster Centroid')
