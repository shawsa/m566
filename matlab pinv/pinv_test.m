A = [1,2,3; 4, 5, 6];
%A = [1,0,;0,1;0,2];

psudo = pinv(A)

A*psudo*A;

psudo' * psudo;
psudo * psudo';

inv(A'*A)*A'

%% 

A = ones(6,3);
b = ones(6,1);

x = A\b

x = pinv(A)*b