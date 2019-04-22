


%%%%%%%%% truncate the ori cqt features %%%%%%%%%%
cqt_truncated = cell(size(cqtFeatureCell));
for i = 1:size(cqtFeatureCell,2)
    if size(cqtFeatureCell{1,i}, 2) < 400 
        if size(cqtFeatureCell{1,i}, 2) > 200
            cqt_truncated{i} = [cqtFeatureCell{1,i} cqtFeatureCell{1,i}(:, 1:400-size(cqtFeatureCell{1,i}, 2))];
        elseif size(cqtFeatureCell{1,i}, 2) > 100
            temp = [cqtFeatureCell{1,i} cqtFeatureCell{1,i} cqtFeatureCell{1,i} cqtFeatureCell{1,i}];
            cqt_truncated{i} = temp(:,1:400);
        else
            temp = [cqtFeatureCell{1,i} cqtFeatureCell{1,i} cqtFeatureCell{1,i} cqtFeatureCell{1,i} cqtFeatureCell{1,i} cqtFeatureCell{1,i} cqtFeatureCell{1,i} cqtFeatureCell{1,i} cqtFeatureCell{1,i}];
            cqt_truncated{i} = temp(:,1:400);
        end
    else
        cqt_truncated{i} = cqtFeatureCell{1,i}(:,1:400);
    end
    i
end

%%%%%%%% normalize the truncated features %%%%%%%%%
b = cell(1,1);
x_train = zeros(3014,863,400);
for i = 1:3014
b{:} = mapminmax(reshape(cqt_truncated(i,:,:), [863 400]));
x_train(i,:,:) = cell2mat(b);
i
end



























