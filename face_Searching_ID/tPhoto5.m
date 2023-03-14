function tPhoto5(void)
% training images (photo)
%clear all; % be heedfull of persistent

N1=40;   % Number of eigenfaces or signatures used for each image, 40 people (20).
M1=10;	 % the number of photos per person (10)

n = M1*N1;	% the number of images (channels)
M = 112; 	% H width
N = 92;		% V length
dimensions = n;

%% Loading the database into matrix v
w=load_database_3(N1, M1);

%x = zeros(n, M*N);	% 45*40=1800
for k=1:n,
%    img = imread(src);
    label_1(k) =k;
    x(k,:) = w(:,k);
    img = w(:,k);
    img1{k} = reshape(img,M,N);
end;

images = reshape(x',M,N,size(x,1));
labels = label_1;
% check image size
[is1, is2, is3] = size(images);
fprintf('image width : %d \n', is1)
fprintf('image Length : %d \n', is2)
fprintf('number of images : %d \n', is3)

figure(11)
subplot(1,2,1)
imshow(images(:,:,1), 'Initialmagnification', 'fit', 'DisplayRange',[]);
title('example original image, the first')
subplot(1,2,2)
imshow(images(:,:,end), 'Initialmagnification', 'fit', 'DisplayRange',[]);
title('example original image, the end')

% data preprocessing
ldata = zeros(size(images, 3), size(images, 1)*size(images, 2));
for k=1:size(images, 3)
	tmp = images(:,:,k);
	ldata(k,:) = tmp(:);
end

% select number of Eigenface to use
dm = 10;    % hard coding(2) or n-1 or n

% Random number selection
pick = round(n*rand(1,1)); % 0<=pick<=dimension(n)
%Q = fix(pick/n);
Q = pick;
fprintf('finding %d \n', Q)
fprintf('Enter\n')

tt_data = ldata(pick, :);
tt_ans = labels(:,pick);
targetfig = reshape((tt_data), is1, is2);
tr_data = ldata([1:pick-1 pick+1:end], :); % you but me
%tr_data = ldata;    % everybody (you and me)
is3 = size(tr_data, 1);

% mean of training data
mu_data = mean(tr_data);

% Remove mean
meansub_data = tr_data - mu_data; % repmat(mu_data, size(tr_data, 1), 1);
x = meansub_data;

figure(12)
imshow(reshape(mu_data,is1,is2), 'Initialmagnification', 'fit', 'DisplayRange',[]);
title('mean face')

% find Eigenface v.2
[U,S,V] = svd(x');
%eig_tr_org = U(:,[1:dm])*V(:,[1:dm])';
%for i=1:1:dm
%  eigen_fcs1=U(:,i)*V(:,i)'; 	% eigenface formulation v.1
%end
eigen_fcs1=U(:,[1:dm])*V([1:dm],[1:dm])'; 	% eigenface formulation v.2

% select svd or bigger Eigenface
eig_tr =  eigen_fcs1;   %eig_tr = sort_v(:, 1:dm);

% find coefficients
coeff_tr = meansub_data*eig_tr;

figure(13)
array_Vlarge=[];
if dm < 9
    dm_d1 = dm;	% dm_d1 figures
elseif dm > 8
    dm_d1 = 9;	% 9 figures
end
%
for i = 1:9	% dm figures max: 9
    if(i>dm_d1)
       tmp_array9{ceil(i/3),mod(i-1,3)+1}(:,:) = zeros(is1, is2);
    else
       tmp_array9{ceil(i/3),mod(i-1,3)+1}(:,:) = reshape(eig_tr(:,i), is1, is2);
    end
end
array_Vlarge=cell2mat(tmp_array9); 
imshow(array_Vlarge ,'Initialmagnification','fit', 'DisplayRange',[]);
str = sprintf('Eigenfaces example first %d among %d', dm_d1, dm);
title(str)

figure(14)
subplot(121)
imshow(targetfig,'Initialmagnification','fit', 'DisplayRange',[]);
str = sprintf('searching... %d', tt_ans);
title(str)

% processing test data
meansub_tt = tt_data - mu_data;
coeff_tt = meansub_tt*eig_tr;

% find answer
subplot(122)
z = zeros(1, is3);
for i = 1:is3
    z(1,i) = norm(coeff_tr(i,:) - coeff_tt, 2);
    imshow(images(:,:,i))
    drawnow;
end

[min_z, min_zi] = min(z);
subplot(122)
imshow(images(:,:,min_zi), 'DisplayRange',[]);
str = sprintf('answer... %d', min_zi);
title(str)
    
% image reconstruction
reco = eig_tr*coeff_tt';
recofig = reco + mu_data';
final_reco = reshape(recofig, is1, is2);

figure(15)
imshow(final_reco, 'Initialmagnification', 'fit', 'DisplayRange',[]);
str = sprintf('reconstructed image used %d eigenface', dm);
title(str)
