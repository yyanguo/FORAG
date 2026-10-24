clc 
clear 
close all 
addpath(genpath('./'));
file_path = './Data';
%
files = dir(strcat(file_path,'/','*.mat'));
datanum = length(files);
resPath = './res_ONMF/';
metric = {'ACC','nmi','Purity','Fscore','Precision','Recall','AR','Entropy'};
Begin_Time = ['Running at *** ' datestr(now)]
for r=1:datanum
   load([file_path,'/',files(r).name],'X','Y');
   DataName=[files(r).name] 
   nclass=length(unique(Y));  nsample=length(Y);   nview=length(X);
   Dimsion=[num2str(nsample) '*' num2str(nclass) '*' num2str(nview)]
   
   matpath = strcat(resPath,DataName);
    txtpath = strcat(resPath,strcat(DataName,'.txt'));
    if (~exist(matpath,'file'))
        mkdir(matpath);
        addpath(genpath(matpath));
    end
    dlmwrite(txtpath, strcat('Dataset:',cellstr(DataName), '  Date:',datestr(now)),'-append','delimiter','','newline','pc');
       for i=1:nview
          X{i} = mapstd(X{i}',0,1);
       end
   k=nclass;
   lambda = [1e-2,1,100,1e4,1e6]; miu=[0.001,0.01,0.1,1,10,100,1000];
   anchor = [k 2*k 3*k];
  for ichor = 1:length(anchor)
        for id = 1:length(lambda)
            for mu=1:length(miu)
            disp(['---Data ',num2str(r),':',DataName,'--Anchor',...
                   num2str(ichor),':',num2str(anchor(ichor)),'--lambda',num2str(id),':',num2str(lambda(id)),'--miu',num2str(mu),':',num2str(miu(mu)),'---']);
            tic;
            [W,A,Z,E,iter,obj,G,F] = alg_FORAG(X,k,anchor(ichor),lambda(id),nclass,nview,nsample,miu(mu));
            [~,idx]=max(F);
            res = Clustering8Measure(Y,idx);
            timer(ichor,id)  = toc;
            fprintf('Anc:%d Res:%12.2f Time:%12.2f \n',[anchor(ichor) res(1) timer(ichor,id)]);
            
            resall{ichor,id} = res;
            objall{ichor,id} = obj;
            
            dlmwrite(txtpath, [anchor(ichor) lambda(id)  res timer(ichor,id)],'-append','delimiter','\t','newline','pc');
            matname = ['_Anchor_',num2str(anchor(ichor)),'_lambda_',num2str(lambda(id)),'_miu_',num2str(miu(mu)),'.mat'];

            save([matpath,'/',matname],'W','A','Z','E','iter','obj','G','F');
            end
        end
  end
  clear resall objall X Y k
end
End_Time = ['Running over *** ' datestr(now)]
