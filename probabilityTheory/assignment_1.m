% 均匀分布生成
% a=-2 b=2 矩阵size为10000 1
size=10000
a=-2
b=2
x =unifrnd (a,b,size,1) ;
y =unifrnd (a,b,size,1) ;
z = x+y;
[f, xi] = ksdensity(z);
subplot(212);
plot(xi, f);
% % 计算方差
avg = z/10000;
var = sum((z - avg).^2)/10000;
disp(var);


