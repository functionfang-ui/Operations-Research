function judge = Check_Sigma(pos,price)
% CHECK_SGEMA calculate the opportunity cost of evey cell.
% Input:
% pos: The dual variable of transportation problem. 
% Be careful, the first m numbers represent ui and the next ten numbers
% represent vj.
% price: The price table which has m rows and n lines.
% Output:
% judge: A matrix with m rows and n lines whose elements represents the opportunity cost of evey cell

% Last modified in 2020.4.24 by Hua Siyuan

[m,n] = size(price);
judge = zeros(m,n);
for i=1:m
    judge(i,:) = price(i,:) - pos(i) - pos(m+1:end);
end
end
