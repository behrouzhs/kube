function [Y, elapsed_time] = KUBE_wrapper(X, no_dim, no_proc, is_cosine, pca_preprocess, no_knn, no_min_knn)
    rng('shuffle');
    if (nargout > 1); tic; end
    
    X = double(X);
    keep_idx = (full(max(X, [], 1) - min(X, [], 1)) ~= 0);
    X = X(:, keep_idx);
    
    if (exist('is_cosine', 'var') == 0 || isempty(is_cosine) || is_cosine == 0)
        if (issparse(X))
            keep_idx = (full(max(X, [], 1)) ~= 0);
            X(:, keep_idx) = bsxfun(@times, X(:, keep_idx), 1 ./ max(X(:, keep_idx), [], 1));
        else
            X = bsxfun(@minus, X, min(X, [], 1));
            X = bsxfun(@times, X, 1 ./ (max(X, [], 1) - min(X, [], 1)));
        end
        
        if (~issparse(X))
            X = bsxfun(@minus, X, mean(X, 1));
        elseif (exist('pca_preprocess', 'var') ~= 0 && isempty(pca_preprocess) == 0 && pca_preprocess > 0)
            X = full(bsxfun(@minus, X, mean(X, 1)));
        end
        
        if (exist('pca_preprocess', 'var') ~= 0 && isempty(pca_preprocess) == 0 && pca_preprocess > 0)
            if (exist('no_dims', 'var') == 0)
                initial_dims = 3;
            else
                initial_dims = no_dim;
            end

%             X = X * randn(size(X, 2), 1000);
            initial_dims = min(size(X, 2), max(50, 5 * initial_dims));
            if (initial_dims < size(X, 2))
                [U, S, ~] = svds(X, initial_dims);
                X = U * S;
            end
        end
    end
    
    if (nargout > 1); disp(['Preprocessing done in ' toc 'seconds']); end
    
    path = which('UBE_c');
    path = fileparts(path);
    
    rnd_num = randi(1000000);
    data_path = ['data_' num2str(rnd_num) '.dat'];
    if (issparse(X) ~= 0)
        [col, row, val] = find(X');
        save_matrix([(row - 1) (col - 1) val], data_path);
    else
        save_matrix(X, data_path);
    end
    
    command = fullfile(path, './ube');
    if (exist('is_cosine', 'var') ~= 0 && isempty(is_cosine) == 0 && is_cosine > 0)
        command = [command ' -c'];
    end
    if (exist('no_dim', 'var') ~= 0 && isempty(no_dim) == 0 && no_dim > 1)
        command = [command ' -d ' num2str(no_dim)];
    end
    if (exist('no_knn', 'var') ~= 0 && isempty(no_knn) == 0 && no_knn > 1)
        command = [command ' -k ' num2str(no_knn)];
    end
    if (exist('no_min_knn', 'var') ~= 0 && isempty(no_min_knn) == 0 && no_min_knn > 1)
        command = [command ' -m ' num2str(no_min_knn)];
    end
    if (exist('no_proc', 'var') ~= 0 && isempty(no_proc) == 0 && no_proc > 1)
        command = [command ' -p' num2str(no_proc)];
    end
    command = [command ' -i ' data_path];
    
    system(command);
    if (nargout > 1); elapsed_time = toc; end
    
    result_path = ['data_' num2str(rnd_num) '_ube.dat'];
    if (exist(result_path, 'file') ~= 0)
        Y = load_matrix(result_path);
%         coeff = pcacov(cov(Y));
%         Y = Y * coeff(:, 1:no_dim);
        delete(result_path);
    else
        disp('Could not compute embedding!');
        Y = [];
    end
    delete(data_path);
end

function save_matrix(X, path)
    [n, d] = size(X);
    h = fopen(path, 'wb');
	fwrite(h, n, 'integer*4');
	fwrite(h, d, 'integer*4');
    fwrite(h, X', 'double');
	fclose(h);
end

function X = load_matrix(path)
    h = fopen(path, 'rb');
	n = fread(h, 1, 'integer*4');
	d = fread(h, 1, 'integer*4');
	X = fread(h, n * d, 'double');
    X = reshape(X, [d n])';
	fclose(h);
end