function [W,A,Z,E,iter,obj,G,F] = alg_FORAG(X,numdim,numanc,lambda,numcla,numview,numsam,miu)
% m      : the number of anchor. the size of Z is m*n.
% X      : n*di

%% initialize
maxIter = 50 ; % the number of iterations

G = eye(numanc,numcla); % m * c
Z = eye(numanc,numsam); % m * n
A = eye(numdim,numanc); % l*m
XX = [];
for p = 1 : numview
    XX = [XX;X{p}];
end
[XU,~,~]=svds(XX',numanc);
rand('twister',12);
[IDX,~] = litekmeans(XU,numanc);
for i = 1:numsam
    Z(IDX(i),i) = 1;
end
F = eye(numcla,numsam); 
parfor i = 1:numview
   di = size(X{i},1); 
   E{i} = Z;
   W{i} = zeros(di,numdim);
end

flag = 1;
iter = 0;
%%
while flag
    iter = iter + 1;
    
    %% optimize Wi  di*m
    parfor iv=1:numview
        TW = X{iv}*(Z+E{iv})'*A';      
        [U,~,V] = svd(TW,'econ');
        W{iv} = U*V';
    end
    clear U V TW
    
    %% optimize A  l*m
    TA=zeros(numdim,numanc);
    for iv=1:numview
        TA = TA + W{iv}'*X{iv}*(Z+E{iv})';       
    end
    [U,~,V] = svd(TA,'econ');
    A = U*V';
    clear U V TA
    
    %% optimize Ei  m*n
    parfor iv=1:numview
        TE = A'*W{iv}'*X{iv}-0.5*Z;      
        for ii=1:numsam
            col_M = TE(:,ii)./(0.5+2*lambda);
            E{iv}(:,ii) = EProjSimplex_new(col_M);
        end
    end
    clear col_M TE

%% optimize Z  m*n
    TZ=zeros(numanc,numsam);
    for iv=1:numview
        TZ = TZ + A'*W{iv}'*X{iv} - 0.5*E{iv};
    end
    BZ=TZ + 2*miu*G*F;
     for ii=1:numsam
        col_N = BZ(:,ii)./(2*miu+0.5*numview);
        Z(:,ii) = EProjSimplex_new(col_N);
     end
    clear TZ BZ col_N
%% optimize G  m*k
    TG=Z*F';
    [U,~,V] = svd(TG,'econ');
    G = U*V';
    clear U V TG
    
    %% optimize F k*n
    F=zeros(numcla,numsam);
    for i=1:numsam
        Dis=zeros(numcla,1);
        for j=1:numcla
            Dis(j)=(norm(Z(:,i)-G(:,j)))^2;
        end
        [~,r]=min(Dis);
        F(r(1),i)=1;
    end
    %%
    term1 = 0;
    term2 = 0;
    for iv = 1:numview
        term1 = term1 + norm(X{iv}-0.5*W{iv}*A*(Z+E{iv}),'fro')^2;
        term2 = term2 + norm(E{iv},'fro')^2;
    end
    term3=norm(Z-G*F,'fro')^2;
    
    obj(iter) = term1+lambda*term2+miu*term3;
    
    if (iter>2) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-4 || iter>maxIter || obj(iter) < 1e-10)
        flag = 0;
    end
end
         
         
    
