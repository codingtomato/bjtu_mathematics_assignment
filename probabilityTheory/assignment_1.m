% ���ȷֲ�����
% a=-2 b=2 ����sizeΪ10000 1
size=10000
a=-2
b=2
x =unifrnd (a,b,size,1) ;
y =unifrnd (a,b,size,1) ;
z = x+y;
[f, xi] = ksdensity(z);
subplot(212);
plot(xi, f);
% % ���㷽��
avg = z/10000;
var = sum((z - avg).^2)/10000;
disp(var);


