import math

index_list = [451, 651]

for env in index_list:
    i = env - 1
    variance = i % 3
    means = math.floor((i % 9) / 3.0)
    change_durations = math.floor((i % 18) / 9.0)
    theta_index = math.floor((i % 378) / 18.0)
    change_count_index = math.floor(i / 378)

    print("Environment {} has i_change_count {}, i_theta {}, i_change_durations {}, i_means {} and i_variance {}".format(env, change_count_index, theta_index, change_durations, means, variance))
