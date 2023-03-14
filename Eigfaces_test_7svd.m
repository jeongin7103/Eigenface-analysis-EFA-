% training images
clear all;
n = 20;		% the number of images (channels)
M = 45; 	% H width
N = 40;		% V length
dimensions = n;

x = zeros(n, M*N);	% 45*40=1800
for k=1:n,
    src = sprintf('./src_img/%d.png',k);
    img = double(rgb2gray(imread(src)));
    %img = imread(src);
    %figure;     imshow(uint8(img));
    x(k,:) = (img(:))';
    img1{k} = reshape(uint8(img),M,N);
end;

idx=1:1:20;
z = make_draw_array(img1,idx);
figure(1),imshow(z,'Initialmagnification','fit'); title('original images')

x=x';	% keep image size (1800)

%%% V.1
% pca
c = cov_2(x);
[v, d] = eig(c);

org_mean = mean(x,1);
meanadjusted = x-repmat(org_mean,size(x,1),1);
%meanadjusted = meanadjust_1(x);

xx1=diag(d);	% max(D)
[xc,xci]=sort(xx1,'descend');		% largest eigenval

%%% V.2
N9=dimensions;
[U,S,V]=svd(x);
Vlarge=U(:,[1:N9])*V([1:N9],[1:N9])';    % Pick the eignevectors corresponding to the 10/20 largest eigenvalues. 
sort_normal_eigs=V;
final_eigvects=sort_normal_eigs(:,[1:N9]);

eigenfaces=[];
for k=1:n
    c  = Vlarge(:,k);
    eigenfaces{k} = reshape(c,M,N);
end

z = make_draw_array(eigenfaces,xci);
figure(2);
%imshow(uint8(z),'Initialmagnification','fit');
imshow(z,'Initialmagnification','fit', 'DisplayRange',[]);
title('eigenfaces')


%%%%% ver 1.1
%finaldata = final_eigvects*meanadjusted';		% coefficients (features extraction: finaldata)
Vlarge = meanadjusted*final_eigvects;		% Big eigenfaces, coefficients (features extraction: finaldata)

% features extraction: linear coefficients of eigenfaces
%for k=1:n,
%    coeff_eigen(k,:) = finaldata*meanadjusted(:,k);		% [a,b]: ax + by = P(x,y)
%end;
coeff_eigen = meanadjusted'*Vlarge;	% linear combination of the eigenvectors

%%%%%%%%%%%%

%final_data = finaldata';

%means = mean(Data,1)';
tmp = Vlarge * final_eigvects';	%	inv(final_eigvects) * finaldata
tmpmeanadjusted = tmp+repmat(org_mean,size(x,1),1);	% add the orgininal mean to the data
originaldata = tmpmeanadjusted;

% recovering the original image using selected eigenfaces
img1=[];
for k=1:n
    img1{k} = reshape(originaldata(:,k),M,N);
end
%
idx=1:1:20;
z = make_draw_array(img1,idx);
figure(3),imshow(uint8(z),'Initialmagnification','fit'); title('recovered images v1.1')

%finaldata = finaleigs'*meanadjusted;

%%%%% ver 1.2
%
coeff_c = meanadjusted*final_eigvects;		% coefficients (features extraction: finaldata)
tmp = coeff_c * final_eigvects';	%	inv(final_eigvects) * finaldata

tmpmeanadjusted = tmp+repmat(org_mean,size(x,1),1);	% add the orgininal mean to the data
originaldata = tmpmeanadjusted;

% recovering the original image using selected eigenfaces
img1=[];
for k=1:n
    img1{k} = reshape(originaldata(:,k),M,N);
end
%
idx=1:1:20;
z = make_draw_array(img1,idx);
figure(4),imshow(uint8(z),'Initialmagnification','fit'); title('recovered images v1.2')

%%%%% ver 1.3
%
final_eigvects = sort_normal_eigs;

Vlarge = meanadjusted*final_eigvects;		% Bigger eigenfaces, coefficients (features extraction: finaldata)
%Vlarge = x*final_eigvects;		% Bigger eigenfaces, coefficients (features extraction: finaldata)
%coeff_eigen = x'*Vlarge;
%tmp = Vlarge * coeff_eigen';
tmp = Vlarge * final_eigvects'; %tmp = final_eigvects*Vlarge'; tmp = tmp'; 
%tmp = Vlarge * final_eigvects;		% freaky faces ?

tmpmeanadjusted = tmp+repmat(org_mean,size(x,1),1);	% add the orgininal mean to the data
originaldata = tmpmeanadjusted;

% recovering the original image using selected eigenfaces
img1=[];
for k=1:n
    img1{k} = reshape(originaldata(:,k),M,N);
end
%
idx=1:1:20;
z = make_draw_array(img1,idx);
figure(13),imshow(uint8(z),'Initialmagnification','fit'); title('recovered images v1.3')

%%%%% ver 1.4
%
final_eigvects = sort_normal_eigs;

Vlarge_r = meanadjusted*final_eigvects;		% Bigger eigenfaces, coefficients (features extraction: finaldata)

%% Calculating the coefficient or signature for each image
%cv=zeros(size(meanadjusted,2),size(Vlarge_r,2));
for i=1:size(meanadjusted,2)
    cv(i,:)=meanadjusted(:,i)'*Vlarge_r;    % Each row in cv is the coefficient or signature for one image.
end

cv1=normc(cv);

%tmp=zeros(size(Vlarge_r));
for i=1:size(cv1,2)
    tmp(:,i) = Vlarge_r * cv1(i,:)';	
end
%tmp = Vlarge_r * cv1';		%tmp = Vlarge_r * final_eigvects';

tmpmeanadjusted = tmp+repmat(org_mean,size(x,1),1);	% add the orgininal mean to the data
originaldata = tmpmeanadjusted;

% recovering the original image using selected eigenfaces
img1=[];
for k=1:n
    img1{k} = reshape(originaldata(:,k),M,N);
end
%
idx=1:1:20;
z = make_draw_array(img1,idx);
figure(14),imshow(uint8(z),'Initialmagnification','fit'); title('recovered images v1.4')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% compression ? %%%%%%%%%%%%%%
%%%%% ver 2.1

rn1 = n-5;		% dimension reduction ?
cv1 = [cv1(:,1:rn1) zeros(n,n-rn1)];

%tmp=zeros(size(Vlarge_r));
for i=1:n
    tmp(:,i) = Vlarge_r * cv1(i,:)';	
end

%tmp = Vlarge_r * cv1';		%tmp = Vlarge_r * final_eigvects';

tmpmeanadjusted = tmp+repmat(org_mean,size(x,1),1);	% add the orgininal mean to the data
originaldata = tmpmeanadjusted;

% recovering the original image using selected eigenfaces
img1=[];
for k=1:n
    img1{k} = reshape(originaldata(:,k),M,N);
end
%
idx=1:1:20;
z = make_draw_array(img1,idx);
figure(5),imshow(uint8(z),'Initialmagnification','fit'); title('recovered images v2.1')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% compression ? %%%%%%%%%%%%%%
%%%%% ver 2.2

rn1 = n-9;		% dimension reduction ?
%reu_eigvectors = [sort_normal_eigs(:,1:rn1) zeros(size(sort_normal_eigs,2),n-rn1)];	% reduction in horizontal direction only
reu_eigvectors = [sort_normal_eigs(:,1:rn1)];	% reduction in horizontal direction only

final_eigvects=reu_eigvectors;
%
coeff_c = meanadjusted*final_eigvects;		% coefficients (features extraction: finaldata)
tmp = coeff_c * final_eigvects';					%	inv(final_eigvects) * finaldata

tmpmeanadjusted = tmp+repmat(org_mean,size(x,1),1);	% add the orgininal mean to the data
originaldata = tmpmeanadjusted;

% recovering the original image using selected eigenfaces
img1=[];
for k=1:n
    img1{k} = reshape(originaldata(:,k),M,N);
end
%
idx=1:1:20;
z = make_draw_array(img1,idx);
figure(6),imshow(uint8(z),'Initialmagnification','fit'); title('recovered images, dimension reduced v2.2')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% compression ? %%%%%%%%%%%%%%
%%%%% ver 2.3

rn1 = n-5;		% dimension reduction ?
reu_eigvectors=[];
reu_eigvectors = [sort_normal_eigs(1:rn1,:); zeros(n-rn1,size(sort_normal_eigs,2))];	% reduction in horizontal direction only

final_eigvects=reu_eigvectors;
coeff_c = meanadjusted*final_eigvects;		% coefficients (features extraction: finaldata)
tmp = coeff_c * final_eigvects';					%	inv(final_eigvects) * finaldata
tmpmeanadjusted = tmp+repmat(org_mean,size(x,1),1);	% add the orgininal mean to the data
originaldata = tmpmeanadjusted;

% recovering the original image using eigenfaces, reduced
img1=[];
for k=1:n
    img1{k} = reshape(originaldata(:,k),M,N);
end
%
idx=1:1:20;
z = make_draw_array(img1,idx);
figure(7),imshow(uint8(z),'Initialmagnification','fit'); title('recovered images, dimension reduced v2.3')



%%%%%%%%%%%%%%%% STOP HERE!!!! %%%%%%%%%%%%%%%%

rn1 = n-8;
meana1 = meanadjusted';
meana2 = meana1(1:rn1,:);
reu_eigvectors = sort_normal_eigs(1:rn1,1:rn1);
tmp1 = reu_eigvectors*meana2;		% coefficients (features extraction: finaldata)

tmp2 = reu_eigvectors' * tmp1;	%	inv(final_eigvects) * finaldata
tmp2=tmp2';
reu_omean=org_mean(1:rn1);
tmpmeanadjusted = tmp2+repmat(reu_omean,size(tmp2,1),1);	% add the orgininal mean to the data
originaldata = tmpmeanadjusted;

% recovering the original image using reduced eigenfaces as rn1
img1=[];
for k=1:n
    img1{k} = zeros(M,N);
end

for k=1:rn1
    img1{k} = reshape(originaldata(:,k),M,N);
end

%
idx=1:1:n;
z = make_draw_array(img1,idx);
figure(27),imshow(uint8(z),'Initialmagnification','fit'); title('recovered images')
