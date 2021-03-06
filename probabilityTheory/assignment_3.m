var = 0;
% 假定A的值
A = 34;
N = [1, 2, 4, 8, 16, 32];
% 存放不同均值下的方差
varArr = (6);
for i = 1:length(N)
    nNum = 0;
    varA = 0;
    % 每种N值对应1000次
    for count = 1:1000
        x = 0;
        for j = 1:N(i)
            wi = randn();
            x = x + A + wi;
        end
        nNum = N(i);
        % 预测的A值
        A_hat = x/nNum;
        varA = varA + (A_hat - A)^2;
    end
    varArr(i) = varA/1000;
end
plot(N, varArr)
