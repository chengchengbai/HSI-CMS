clear,clc,close all;
addpath(genpath('Utilities'));

time0 = clock;
%% Read data
name = 'Fake_and_real_lemons';
dataZ = load(['..\HSRLRT\data\dataZof',name,'.mat']);
RZ = dataZ.dataZ;
RZ = im2double(RZ);

rzSize = size(RZ);

sf = 32;     % scaling factor
sz = [rzSize(1),rzSize(2)];

kernel_type = 'Uniform_blur';
par = ParSet_new(sf,sz,kernel_type);

P = create_P();         % P: Y = PZ from WeishengDong 
RZ2d = loadHSI(RZ);
X = par.H(RZ2d);      % down sampling from Z
H = create_H(sz,sf);    % H down sampling factor
% X = RZ2d*H;
Y = P*RZ2d;

% kernel_type = 'Uniform_blur';
% par = ParSet_new(sf,sz,kernel_type);

% base line use imresize bicubic iterpolation
Z = base_bicubic(X,sf);    
% Z = alternating_back_projection(Z,X,Y,P,H);

[PSNR_base,RMSE_base] = Evaluate(RZ2d, Z);

Z3d = ReshapeTo3D(Z,rzSize);
Y3d = ReshapeTo3D(Y,[rzSize(1),rzSize(2),3]);
I_1 = eye(size(P,2));

%% Outer loop
V1 = zeros(rzSize(3),rzSize(1)*rzSize(2));
V2 = zeros(rzSize(3),rzSize(1)*rzSize(2));

XHT = par.HT(X);  
XHT1 = X*H';

% save results and parameters
date = datestr(now());
date(date==':')='-';
fp=fopen(['exp-record\',name,date,'.txt'],'w');    
fprintf(fp,'DataSet : %s \n\n',name);
fprintf(fp,'Raw image size rzSize : %d * %d * %d\n\t',rzSize(1),rzSize(2),rzSize(3));
fprintf(fp,' sf : %d \n\n',sf);
writePar2File(fp,par);
fprintf(fp,'Bicubic baseline:\n\t');
fprintf(fp,'PSNR_base: %9.5f \n\t',PSNR_base);
fprintf(fp,'RMSE_base: %9.5f \n\n',RMSE_base);
fprintf(fp,'This Result :\n\t');

tic;
[Wi,index]= getWeightPca(Y3d,par);       % get W from Y-RGB
t1 = toc;
fprintf(fp,'getWeightPca time comsuming: %9.5f \n\n',t1);

Groups = cell(1,par.nCluster);
Li = cell(1,par.nCluster);

fprintf('Iteration 0 PSNR_base: %9.5f \n',PSNR_base);
fprintf('Iteration 0 RMSE_base: %9.5f \n\n',RMSE_base);

PSNR_last = PSNR_base;
to = 1;
for t = 1 : par.iter

    uGrpPatchs = Im2Patch3D(Z3d,par);                    
    sizeLi = size(uGrpPatchs);
    
    tic;
    for i = 1:par.nCluster
        Groups{i} = uGrpPatchs(:,:,index{i});      % Groups{i}is an cluster
    end
    t2 = toc;

    tic
    for i = 1 : par.nCluster
        tempLi = ReshapeTo2D_C(Groups{i});        %  3-mode product
        resLi = tempLi*Wi{i};
        Li{i} = ReshapeTo3D(resLi',size(Groups{i}));
%           Li{i} = Groups{i};
    end
    t3 = toc;

    tic;
    Epatch          = zeros(sizeLi);
    W               = zeros(sizeLi(1),sizeLi(3));
    for i = 1:par.nCluster
        Epatch(:,:,index{i})  = Epatch(:,:,index{i}) + Li{i};
        W(:,index{i})         = W(:,index{i})+ones(size(Li{i},1),size(Li{i},3));
    end
    [L, ~]  =  Patch2Im3D( Epatch, W, par, rzSize);              % recconstruct the estimated MSI by aggregating all reconstructed FBP goups.
    t4 = toc;
    
    L = ReshapeTo2D(L);   

    tic;
    U  = ((par.mu+par.eta)^-1)*(par.mu*Z+par.eta*L+V1/2);        % Update splitting variables U with Eq.15
    S  = ((P'*P+par.mu*I_1)^-1)*(P'*Y+par.mu*Z+V2/2);            % I denotes an identity matrix with proper size.Update splitting variables U with Eq.17        
    t5 = toc;
    
    tic;
    % Eq.19
    B = (XHT+par.mu*U+par.mu*S-V1/2-V2/2)';
    for j = 1: rzSize(3)
        [z,flag]     =    pcg( @(x)A_x(x, par.mu, par.fft_B, par.fft_BT, sf, sz), B(:,j), 1E-3, 350, [], [], Z(j, :)' );
        Z(j, :)      =    z';
    end
    % end of Weisheng Dong
    t6 = toc;
    
    V1 = V1+2*par.mu*(Z-U);                                  % Update multipliers V1 and V2 as Eq.20;
    V2 = V2+2*par.mu*(Z-S);
    par.mu = par.rho*par.mu;

    [PSNR, RMSE] = Evaluate(RZ2d, Z);
%     if mod(t,5)==0||t>=15
%         fprintf('Iteration %d PSNR: %9.5f \n',t,PSNR);
%         fprintf('Iteration %d RMSE: %9.5f \n\n',t,RMSE);
%         fprintf(fp,'Iteration %d PSNR: %9.5f \n',t,PSNR);
%         fprintf(fp,'Iteration %d RMSE: %9.5f \n\n',t,RMSE);
        if PSNR - PSNR_last <0.006
            to = to + 1;
        else
            to = 0;
        end
%     end

    if to>3
        break;
    end
    
    PSNR_last = PSNR;
    Z3d = ReshapeTo3D(Z,rzSize);
end

%% Trick for tail processing
Z = alternating_back_projection(Z,X,Y,P,H);
[PSNR, RMSE] = Evaluate(RZ2d, Z);
fprintf('The final PSNR: %9.5f \n',PSNR);
fprintf('The final RMSE: %9.5f \n',RMSE);
fprintf(fp,'The final PSNR: %9.5f \n',PSNR);
fprintf(fp,'The final RMSE: %9.5f \n\n',RMSE);
fprintf(fp,'Total time comsuming = %f min\n', (etime(clock,time0)/60));
fclose(fp);
