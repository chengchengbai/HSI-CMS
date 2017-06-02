function [PSNR, RMSE] = Evaluate(R, Z)
% R: raw image
MSE             =    mean( mean( (R-Z).^2 ) );
PSNR            =    10*log10(1/MSE);
RMSE            =    sqrt(MSE)*255;
