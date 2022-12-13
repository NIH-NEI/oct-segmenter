function max_height = crop_oct_images(input_dir, output_dir, ...
    annot_output_dir, csv_input_dir ...
)
    %
    % Using a 'threshold' value, it estimates the location of the ILM
    % layer. Then it fits a second degree polynomial to the estimated
    % coordinates. Finally, it uses the min and max curve values and
    % fixed top and bottom margins to crop the OCT image and remove as
    % much background as possible
    %
    % Note that the 'threshold' value used here works with OCT images that
    % have been processed using the 'exponentiation + compensation' method
    % described in:
    %
    % Girard MJ, Strouthidis NG, Ethier CR, Mari JM. Shadow removal and
    % contrast enhancement in optical coherence tomography images of the
    % human optic nerve head. Invest Ophthalmol Vis Sci. 2011;52(10):7738-7748.
    % Published 2011 Sep 29. doi:10.1167/iovs.10-6925
    %
    % Note: This function requires the MATLAB "Computer Vision Toolbox"
    %
    % Input
    % -----
    % [string]
    % input_dir: Path to the input directory. The TIFF files found will be
    % processed.
    %
    % [string]
    % output_dir: Output path to save the cropped TIFF images.
    %
    % [string]
    % annot_output_dir: Output path to save the input images with the
    % estimates of the ILM (yellow) and the fitted polynomial (green)
    %
    % [string]
    % csv_input_dir: Input path to the CSV masks
    %
    % $Author: Bruno Alvisio (bruno@bioteam.net)
    %
    thresholds = [40, 30, 50];
    curvature_threshold = 40;
    crop_top = 30;
    crop_bottom = 200;
    max_height = -Inf;
    top_black_band = 70; % To prevent from noise to be confused as the IPL

    disp("Cropping TIFFs and CSV masks");
    for input_image = dir(input_dir + "/*.tiff")'
        input_image_name = input_image.folder + "/" + input_image.name;
        img = imread(input_image_name);
        ilm = NaN(1, size(img, 2));

        found_ipl = false;
        for threshold = thresholds
            %disp("Finding IPL with threshold " + threshold + ...
            %    " of TIFF: " + input_image_name ...
            %)
            for j = 1:size(img, 2)
                ilm(j) = find(img(top_black_band:end,j) >= threshold, ...
                    1, 'first') + top_black_band;
            end

            img_width = length(ilm);
            x = 1:img_width;
            % Remove hgh-frequency noise (i.e. jumps in data)
            smooth = smoothdata(ilm, "movmedian", 200);
            % Fit polynomial
            p = polyfit(x, smooth, 2);
            ilm_fit = round(polyval(p, x));

            % Get a measure of the difference between the polynomial at the
            % center and sides of the image. If the difference is too large
            % try using another threshold for finding the IPL.
            if abs(ilm_fit(fix(img_width/2)) - ilm_fit(1)) > curvature_threshold ...
                || abs(ilm_fit(fix(img_width/2)) - ilm_fit(img_width)) > curvature_threshold
                disp("WARNING: The polynomial fit has a curvature " + ...
                    "greater than expected for image: " + ...
                    input_image_name + ". Trying with another " + ...
                    "threshold." ...
                );
                continue;
             end

            img_height = size(img, 1);
            new_top = max(1, min(ilm_fit) - crop_top);
            new_bottom = min(img_height, max(ilm_fit) + crop_bottom);
            found_ipl = true;
            new_height = new_bottom - new_top;
            if new_height > max_height
                max_height = new_height;
            end

            cropped_img = img(new_top : new_bottom, :);
            output_image_name = output_dir + "/" + input_image.name;
            imwrite(cropped_img, output_image_name);

            % Interleave x and ILM
            x_t = x(:).';
            ilm_t = ilm(:).';
            x_ilm_concat = [x_t; ilm_t]; % concatenate them vertically
            x_ilm_interleaved = x_ilm_concat(:);

            % Interleave x and fitted polynomial
            ilm_fit_t = ilm_fit(:).';
            x_ilm_fit_concat = [x_t; ilm_fit_t];
            x_ilm_fit_interleaved = x_ilm_fit_concat(:);

            % Interleave smooth
            smooth_t = smooth(:).';
            smooth_concat = [x_t; smooth_t];
            smooth_interleaved = smooth_concat(:);

            img = insertShape(img, "line", ...
                [x_ilm_interleaved'; x_ilm_fit_interleaved'; ...
                smooth_interleaved'], ...
                "Color", ["yellow", "green", "red"] ...
            );

            imwrite(img, annot_output_dir + "/" + input_image.name);

            if exist(csv_input_dir)
                [~, file_name, ~] = fileparts(input_image.name);
                input_csv_file_name = csv_input_dir + "/" + ...
                    file_name + ".csv";
                %disp("Cropping CSV: " + input_csv_file_name);
                mask = readmatrix(input_csv_file_name);
                mask = mask(new_top : new_bottom, :);
                output_csv_file_name = output_dir + "/" + file_name + ...
                    ".csv";
                writematrix(mask, output_csv_file_name);
            end
            break;
        end % end threshold
        if ~found_ipl
            disp("WARNING: Couldn't find IPL layer after trying all " + ...
                "thresholds for image: " + input_image_name ...
            );
        end
    end
end
