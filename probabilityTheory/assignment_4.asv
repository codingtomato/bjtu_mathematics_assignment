% 预测次数的数组
N = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000];
% 随机取-1 1的数组
xArr=[-1,1]
% 误码率数组
BER = (10);
for i = 1:length(N)
    % 预测命中次数
    hits_count = 0;
    % 预测的次数N(i)
    for j = 1:N(i)
        % 随机生成1或-1
        x = xArr(randi(2))
        h = 0.001;
        sum = 0;
        % Recovery X
        for count = 1:N(i)
            y = h*x + randn();
            sum = sum + y;
        end
        x_hat = sum/h*N(i);
        if x_hat * x > 0
            hits_count = hits_count + 1;
        end
    end
    BER(i) = (N(i)-hits_count)/N(i);
end
plot(212)
plot(N, BER);
