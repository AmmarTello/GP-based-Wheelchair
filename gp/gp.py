import numpy as np

from LIRU_GP2 import Train, Predict


class GP:

    @staticmethod
    def train(L0, Sigma0, training_descriptors, training_velocities):
        L, Sigma, K, C, InvC, elapsed_time = \
            Train(L0, Sigma0, training_descriptors, training_velocities, len(training_descriptors))

        return {"L": L, "Sigma": Sigma, "K": K, "C": C, "InvC": InvC, "elapsed_time": elapsed_time}

    @staticmethod
    def test(training_descriptors, testing_descriptors, trained_gp_dict, training_velocities):
        Y_StarMean = np.empty(len(testing_descriptors))  # mean of GP predictions
        Y_StarStd = np.empty(len(testing_descriptors))  # std of GP predictions
        elapsed_time = np.empty(len(testing_descriptors))

        for desc_index, descriptor in enumerate(testing_descriptors):
            Y_StarMean[desc_index], Y_StarStd[desc_index], elapsed_time[desc_index] = \
                Predict(training_descriptors, descriptor, trained_gp_dict["L"], trained_gp_dict["Sigma"],
                        training_velocities, trained_gp_dict["K"], trained_gp_dict["C"], trained_gp_dict["InvC"],
                        len(training_descriptors))

        return {"Y_StarMean": np.array(Y_StarMean), "Y_StarStd": np.array(Y_StarStd),
                "elapsed_time": np.array(elapsed_time)}

    @staticmethod
    def save(file_path, exp_config, train_gp_dict, test_gp_dict,
             train_dataset_dict, test_dataset_dict, metrics):

        np.savez(file_path, exp_config=exp_config, trained_gp_dict=train_gp_dict,
                 test_gp_dict=test_gp_dict, train_dataset_dict=train_dataset_dict,
                 test_dataset_dict=test_dataset_dict, metrics=metrics)

    @staticmethod
    def load(file_path):

        return np.load(file_path, allow_pickle=True)
