var = 0;
% �ٶ�A��ֵ
A = 34;
N = [1, 2, 4, 8, 16, 32];
% ��Ų�ͬ��ֵ�µķ���
varArr = (6);
for i = 1:length(N)
    nNum = 0;
    varA = 0;
    % ÿ��Nֵ��Ӧ1000��
    for count = 1:1000
        x = 0;
        for j = 1:N(i)
            wi = randn();
            x = x + A + wi;
        end
        nNum = N(i);
        % Ԥ���Aֵ
        esA = x/nNum;
        varA = varA + (esA - A)^2;
    end
    varArr(i) = varA/1000;
end
plot(N, varArr)