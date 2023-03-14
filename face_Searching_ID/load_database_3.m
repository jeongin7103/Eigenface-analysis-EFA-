function out=load_database_3(N, M)
% We load the database the first time we run the program.

persistent loaded;
persistent w;

if(isempty(loaded))
    %v=zeros(10304,400);
    v=zeros(10304,N*M	);
    for i=1:N		% 40
    	%src = sprintf('s%d',i);
        %strcat('s',num2str(i));
        %cd(strcat('database\s',num2str(i)));
        for j=1:M 	%10
            %a=imread(strcat(num2str(j),'.pgm'));
            src = sprintf('../face_recognition40svd/src_img/s%d/%d.pgm',i,j);
            a = imread(src);
            v(:,(i-1)*10+j)=reshape(a,size(a,1)*size(a,2),1);
         end
    end
    w=v; % Convert to unsigned 8 bit numbers to save memory. 
    
    loaded=1;  % Set 'loaded' to aviod loading the database again. 
    %
    figure(1);
    for i=1:size(w,2)
	subplot(5,4,rem(i,20)+1);
	% imshow(reshape(uint8(w(:,i)),112,92),'Initialmagnification','fit');
	imshow(reshape(w(:,i),112,92),'Initialmagnification','fit', 'DisplayRange',[]);
	drawnow;
    end
end

out=w;
