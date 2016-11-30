function [dataset inp_size out_size] = dataset_narma_10(input_dt, maxtime, nr_samples)
%DATASET_NARMA_10 - generate Jaeger's 10-th order NARMA data
%  [dataset inp_size out_size] = dataset_narma_10(input_dt, maxtime, nr_samples)
%  Generates a 10-th order NARMA time series of length maxtime seconds, 
%  with a sampling period of input_dt seconds.
%  Adaptation from code by H. Jaeger

% WARNING: This dataset has a flaw !! Long sequences always blow up
% (27/08/08) Hack implemented: keep generating new data until the dataset is
% valid (max <=2)


inp_size=1;
out_size=1;
len = round(maxtime/input_dt);
systemorder = 10;

dataset = struct('inputs', cell(1,nr_samples), 'outputs', cell(1,nr_samples));

for i = 1:nr_samples
    data_OK = false;
    while(~data_OK)
        %Create random input sequence
        dataset(i).inputs = rand(1,len)*.5;
        dataset(i).outputs = 0.1*ones(1,len);
        for n=systemorder+1:len-1
            %Insert suitable NARMA equation on r.h.s.
            dataset(i).outputs(n+1) = .3*dataset(i).outputs(n) + .05*dataset(i).outputs(n)*sum(dataset(i).outputs(n-9:n)) + 1.5*dataset(i).inputs(n-9) * dataset(i).inputs(n) + .1;
        end
        if(sum(isinf(dataset(i).outputs))==0)
            if(sum(dataset(i).outputs>2)==0)
                data_OK=true;
            end
        end
    end
end
