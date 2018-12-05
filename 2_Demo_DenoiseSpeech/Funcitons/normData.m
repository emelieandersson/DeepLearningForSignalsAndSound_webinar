function [predictorsNew,targetsNew] = normData(predictors,targets)

%Normalize and reshape

predictors    = cat(3, predictors{:});
targets       = cat(2, targets{:});
noisyMean     = mean(predictors(:));
noisyStd      = std(predictors(:));
predictors(:) = (predictors(:)-noisyMean)/noisyStd;
cleanMean     = mean(targets(:));
cleanStd      = std(targets(:));
targets(:)    = (targets(:)-cleanMean)/cleanStd;

predictorsNew  = reshape(predictors,size(predictors,1),size(predictors,2),1,size(predictors,3));
targetsNew     = reshape(targets,1,1,size(targets,1),size(targets,2));
end

