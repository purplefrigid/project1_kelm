% figure
% plot(1: 97, T_test(:,4), 'r-*', 1: 97, T_sim2(:,4), 'b-o', 'LineWidth', 1)
% xlabel('预测样本')
% ylabel('预测结果')
% xlim([1, 97])
% grid
% 生成示例数据  
% data = rand(100, 97); % 生成100个样本，每个样本有5个特征  
% 
% % 数据标准化  
% data_mean = mean(data);  
% data_centered = data - data_mean; % 去均值  
% 
% % 计算协方差矩阵  
% cov_matrix = cov(data_centered);  
% 
% % 计算特征值和特征向量  
% [eigen_vectors, eigen_values] = eig(cov_matrix);  
% 
% % 提取特征值和特征向量  
% eigen_values = diag(eigen_values);  
% [~, idx] = sort(eigen_values, 'descend'); % 按特征值降序排序  
% eigen_vectors = eigen_vectors(:, idx); % 重新排序特征向量  
% 
% % 选择前k个主成分  
% k = 80; % 降维到2维  
% W = eigen_vectors(:, 1:k); % 选择前k个特征向量  
% 
% % 数据降维  
% data_reduced = data_centered * W; % 降维后的数据  
% 
% % 数据恢复  
% data_recovered = data_reduced * W' + data_mean; % 恢复数据  
% 
% % 可视化结果  
% figure;  
% subplot(1, 2, 1);  
% scatter3(data(:, 1), data(:, 2), data(:, 3), 'filled');  
% title('原始数据');  
% xlabel('特征1');  
% ylabel('特征2');  
% zlabel('特征3');  
% 
% subplot(1, 2, 2);  
% scatter3(data_recovered(:, 1), data_recovered(:, 2), data_recovered(:, 3), 'filled');  
% title('恢复后的数据');  
% xlabel('特征1');  
% ylabel('特征2');  
% zlabel('特征3');
% 生成示例数据  
clear;
% num_samples = 200;  
% num_features = 10;  
% data = rand(num_samples, num_features); % 生成随机数据  
% labels = randi([0, 1], num_samples, 1); % 随机生成二分类标签  
% 
% % 数据标准化  
% data_mean = mean(data);  
% data_centered = data - data_mean;  
% 
% % PCA降维  
% cov_matrix = cov(data_centered);  
% [eigen_vectors, eigen_values] = eig(cov_matrix);  
% eigen_values = diag(eigen_values);  
% [~, idx] = sort(eigen_values, 'descend');  
% eigen_vectors = eigen_vectors(:, idx);  
% 
% % 选择前k个主成分  
% k = 2; % 降维到2维  
% W = eigen_vectors(:, 1:k);  
% data_reduced = data_centered * W;  
% 
% % 主动学习初始化  
% num_initial_samples = 10; % 初始标注样本数量  
% initial_indices = randperm(num_samples, num_initial_samples);  
% labeled_data = data_reduced(initial_indices, :);  
% labeled_labels = labels(initial_indices);  
% 
% % 主动学习循环  
% num_iterations = 5; % 迭代次数  
% for iter = 1:num_iterations  
%     % 训练简单模型（例如，逻辑回归）  
%     model = fitglm(labeled_data, labeled_labels, 'Distribution', 'binomial');  
% 
%     % 计算未标注样本的预测概率  
%     unlabeled_indices = setdiff(1:num_samples, initial_indices);  
%     unlabeled_data = data_reduced(unlabeled_indices, :);  
%     predictions = predict(model, unlabeled_data);  
% 
%     % 选择最不确定的样本（例如，选择预测概率最接近0.5的样本）  
%     uncertainty = abs(predictions - 0.5);  
%     [~, uncertain_indices] = sort(uncertainty, 'descend');  
%     new_sample_index = unlabeled_indices(uncertain_indices(1)); % 选择最不确定的样本  
% 
%     % 更新标注数据  
%     labeled_data = [labeled_data; data_reduced(new_sample_index, :)];  
%     labeled_labels = [labeled_labels; labels(new_sample_index)];  
%     initial_indices = [initial_indices'; new_sample_index]'; % 更新已标注样本索引  
% end  
% 
% % 可视化结果  
% figure;  
% gscatter(data_reduced(:, 1), data_reduced(:, 2), labels);  
% hold on;  
% scatter(labeled_data(:, 1), labeled_data(:, 2), 100, 'r', 'filled'); % 标注样本  
% title('主动学习与数据降维');  
% xlabel('主成分1');  
% ylabel('主成分2');  
% legend('未标注样本', '标注样本');  
% hold off;
% 设置随机数种子  
rng(1);  

% 加载MNIST数据集  
[xTrain, ~] = digitTrain4DArrayData; % 训练数据，28x28x1的图像  

% 生成器网络  
generator = [  
    imageInputLayer([1 1 100]) % 输入层，100维随机噪声  
    fullyConnectedLayer(256)   % 全连接层  
    reluLayer                   % ReLU激活层  
    fullyConnectedLayer(512)    % 全连接层  
    reluLayer                   % ReLU激活层  
    fullyConnectedLayer(1024)   % 全连接层  
    reluLayer                   % ReLU激活层  
    fullyConnectedLayer(28*28)  % 输出层，生成28x28图像  
    tanhLayer                   % Tanh激活层  
%     reshapeLayer([28 28 1])     % 重塑为28x28x1的图像  
];  

% 判别器网络  
discriminator = [  
    imageInputLayer([28 28 1])  % 输入层，28x28图像  
    convolution2dLayer(5, 32, 'Stride', 2, 'Padding', 'same') % 卷积层  
    leakyReluLayer(0.2)          % Leaky ReLU激活层  
    convolution2dLayer(5, 64, 'Stride', 2, 'Padding', 'same') % 卷积层  
    leakyReluLayer(0.2)          % Leaky ReLU激活层  
    flattenLayer()               % 展平层  
    fullyConnectedLayer(1)       % 全连接层  
    sigmoidLayer                 % Sigmoid激活层  
];  

% 训练参数  
numEpochs = 10000;  
batchSize = 128;  
learningRate = 0.0002;  
numBatches = floor(size(xTrain, 4) / batchSize); % 计算批次数  

% 优化器选项  
options = trainingOptions('adam', ...  
    'MaxEpochs', numEpochs, ...  
    'MiniBatchSize', batchSize, ...  
    'InitialLearnRate', learningRate, ...  
    'Verbose', false, ...  
    'Plots', 'training-progress');  

% 训练GAN  
for epoch = 1:numEpochs  
    for batch = 1:numBatches  
        % 生成随机噪声  
        noise = randn(batchSize, 100);  
        
        % 生成假图像  
        fakeImages = predict(generator, noise);  
        
        % 从真实数据集中获取真实图像  
        realImages = xTrain(:,:,:, (batch-1)*batchSize+1:batch*batchSize);  
        
        % 训练判别器  
        % 真实图像标签为1，假图像标签为0  
        labelsReal = ones(batchSize, 1);  
        labelsFake = zeros(batchSize, 1);  
        
        % 训练判别器  
        % 这里需要实现判别器的训练过程  
        % 使用训练数据和标签更新判别器  
        
        % 训练生成器  
        % 生成器的目标是让判别器认为生成的图像是真实的  
        % 这里需要实现生成器的训练过程  
    end  
end