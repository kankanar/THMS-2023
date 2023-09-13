function [U,V] = structural_nmf(X,UA,k,i)
    i
    U = randi(100,size(X,1),k);
    V = randi(100,k,size(X,2));
    M = 0;
    P = 0;
    mu = 0.0001;
    gama = 0.01;
    ro = 1.2;
    maxiter = 100;
    UjUt = 0;
    for j=1:numel(UA)
        if(j~=i)
           UjUt = UjUt+ UA{j}*UA{j}'; 
        end  
    end
    for j=1:maxiter
    err = Inf;
    %while err>0.001
        U = max(0,(2*X*V' + mu*P - M)/(2*(V*V') + mu*eye(size(V*V'))));
        V = max(0,(U'*U + 0.001*eye(size(U'*U)))\(U'*X));
        P = (2*gama*UjUt + mu*eye(size(UjUt)))\(mu*U + M);
        M = M + mu*(U - P);
        mu = min(10^5,ro*mu);
        err = norm(X - U*V)/norm(X)
    end
end