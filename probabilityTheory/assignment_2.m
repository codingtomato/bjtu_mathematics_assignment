z = 0;
size=10000
a=-2
b=2
for i = 1:100
    z = z + unifrnd (a,b,size,1);
end
[f, xi] = ksdensity(z);
subplot(212);
plot(xi, f);
% º∆À„∑Ω≤Ó
avg = z/10000;
var = sum((z - avg).^2)/10000

