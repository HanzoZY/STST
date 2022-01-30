
%%%%%% TO MODIFY

root_database = 'C:\Users\DeSmedt\Documents\S_DHG2016_sequences\';

idx_gesture = 1;
idx_finger = 1;
idx_subject = 1;
idx_essai = 1;

%%%%%%%%%%%

% Idx of bones in the hand skeleton to display it.

bones = ...
    [
    0     1;
    0     2;
    2     3;
    3     4;
    4     5;
    1     6;
    6     7;
    7     8;
    8     9;
    1    10;
    10    11;
    11    12;
    12    13;
    1    14;
    14    15;
    15    16;
    16    17;
    1    18;
    18    19;
    19    20;
    20    21;
    ];

% Path of the gesture

path_gesture = [root_database 'gesture_' num2str(idx_gesture) '\finger_' num2str(idx_finger) '\subject_' num2str(idx_subject) '\essai_' num2str(idx_essai) '\'];

% Path of the skeleton in the image

path_skeletons_image = [path_gesture,'skeletons_image.txt'];

if exist(path_skeletons_image, 'file') == 2
        
    % Import of the skeleton
    
    skeletons_image = importdata(path_skeletons_image);
    
    % Import of the depth images
    
    pngDepthFiles = zeros(size(skeletons_image,1), 480, 640);
    
    for i = 0:size(pngDepthFiles,1)-1
        pngDepthFiles(i+1,:,:) = importdata([path_gesture num2str(i) '_depth.png']);
    end
    
    skeletons_display = zeros(size(skeletons_image,1), 2, 2, 21);
    
    for idx_skeleton = 1:size(skeletons_image,1)
        
        ske = skeletons_image(idx_skeleton,:);
        
        x = zeros(2, size(bones,1));
        y = zeros(2, size(bones,1));
        
        for idx_bones = 1:size(bones,1)
            
            joint1 = bones(idx_bones, 1);
            joint2 = bones(idx_bones, 2); 
            
            pt1 = ske(1, joint1*2+1:joint1*2+2);
            pt2 = ske(1,joint2*2+1:joint2*2+2);
            
            x(1,idx_bones) = pt1(1,1);
            x(2,idx_bones) = pt2(1,1);
            y(1,idx_bones) =  pt1(1,2);
            y(2,idx_bones) = pt2(1,2);  
        end
        
        % Create skeleton from a skeleton and bones information
        
        skeletons_display(idx_skeleton, 1, : , :) = x;
        skeletons_display(idx_skeleton, 2, : , :) = y;
    end
    
    disp('-------');
    disp(path_gesture);
    
    
    % Display
    
    for j = 1:size(pngDepthFiles,1)
        t = double(squeeze(pngDepthFiles(j,:,:)));
        imagesc(t);
        
        line(squeeze(skeletons_display(j, 1, : , :)), squeeze(skeletons_display(j, 2, : , :)),'LineWidth',2.5);
        
        
        pause(0.03)
    end
    
    
    
    
else
    disp(strcat('There is no gesture in the path: ', path_gesture));
end

