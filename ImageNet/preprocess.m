%pre-process ImageNet dataset :
% 1. convert to grayscale
% 2. resize to 144xN (N >= 128)

main_dir = '~/data/ImageNet';
set = 'train';
output_dir = '~/data/ImageNet_preprocessed';
target_size = 128; % min side

switch set
    case 'val'
        images = dir(fullfile(main_dir, set, '*.JPEG'));
        images = {images.name};
    case 'train'
        class_folders = dir(fullfile(main_dir, set));
        class_folders = class_folders(3:end);
        images = {};
        for i_class = 1:length(class_folders)
            files = dir(fullfile(main_dir, set, ...
                class_folders(i_class).name, '*.JPEG'));
            files = {files.name};
            files = cellfun(@(x)(fullfile(class_folders(i_class).name, x)), files, 'UniformOutput', false);
            images = [images, files];
        end
end

means = zeros(length(images), 1);
errors = zeros(length(images), 1);
parfor i_image = 1:length(images)
    input_path = fullfile(main_dir, set, images{i_image}); 
    output_path = fullfile(output_dir, set, images{i_image});
    %if exist(output_path, 'file')
    %    continue;
    %end
    if ~exist(fileparts(output_path), 'dir')
        mkdir(fileparts(output_path))
    end
    
    try
        a = imfinfo(input_path);
        % convert CMYK to sRGB
        if strcmp(a.ColorType, 'CMYK')
            [status, cmdout] = system(sprintf('convert %s -colorspace sRGB %s', input_path, input_path));
            if status ~= 0
                error(cmdout);
            end
        end
        
        %
        im = imread(input_path);
        if size(im, 3) == 3
            im = rgb2gray(im);
        end

        % resize
        [h, w] = size(im);
        im_size_min = min([h w]);
        im_size_max = max([h w]);
        im_scale = double(target_size) / im_size_min;
        im = imresize(im, im_scale);
        imwrite(im, output_path);

        % collect mean information
        means(i_image) = mean(im(:));
    catch me
        errors(i_image) = 1;
    end
end
fprintf('mean value = %.4f\n', mean(means));