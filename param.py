def _assignment(flag=None):
    if flag is None:
        _assignment.flag = _assignment.flag + 1 if hasattr(_assignment, "flag") else 0+1
    else:
        _assignment.flag = flag + 1
    return _assignment.flag - 1

####################### common ######################
######### direction
DIRECTION_LEFT = _assignment(0)
DIRECTION_RIGHT = _assignment()

####################### perception ######################
# _i = 0
PERCEPTION_BOX = _assignment(0)
PERCEPTION_OCC = _assignment()

####################### prediction ######################
# _i = 0
PREDICTION_LINEAR = _assignment(0)


####################### output ######################
# _i = 0
OUTPUT_TRAJECTORY = _assignment(0)
OUTPUT_CONTROL = _assignment()

####################### constraint ######################
# _i = 0
CONSTRAINT_COLLISION = _assignment(0)
# cartesian
CONSTRIANT_SPEED_UB = _assignment()
CONSTRIANT_SPEED_LB = _assignment()
CONSTRIANT_ACCELERATION = _assignment()
CONSTRIANT_DECELERATION = _assignment()
CONSTRIANT_JERK = _assignment()
CONSTRIANT_CURVATURE = _assignment()
CONSTRIANT_HEADING = _assignment()
CONSTRIANT_STEER = _assignment()


# frenet
CONSTRIANT_LATERAL_OFFSET = _assignment()
CONSTRIANT_LATERAL_VELOCITY = _assignment()
CONSTRIANT_LATERAL_ACCELERATION = _assignment()
CONSTRIANT_LATERAL_JERK = _assignment()

CONSTRIANT_LONGITUDINAL_PROGRESS = _assignment()
CONSTRIANT_LONGITUDINAL_VELOCITY = _assignment()
CONSTRIANT_LONGITUDINAL_ACCELERATION = _assignment()
CONSTRIANT_LONGITUDINAL_JERK = _assignment()

####################### collision ######################
# for collision_checker
COLLISION_CHECKER_SAT = _assignment(0)
COLLISION_CHECKER_AABB = _assignment()
COLLISION_CHECKER_DISK = _assignment()
COLLISION_CHECKER_OCC = _assignment()

####################### DATA ######################
# for data closed-loop engine
DATASET_FORMAT_JSON = _assignment(0) # 将log buffer以json格式存储
DATASET_FORMAT_TENSOR = _assignment() # 将log buffer以tensor格式存储
DATASET_FORMAT_RAW = _assignment() # 将log buffer以原始数据格式存储

LOAD_TYPE_DATALOADER = _assignment(0) # 读取数据 并 转化为dataloader
LOAD_TYPE_RAW = _assignment() # 读取数据 并 保持原始数据格式


####################### RL ######################
# NN MODE FLAG
NN_TRAIN_MODE = _assignment(0)
NN_EVAL_MODE = _assignment()

####################### interface ######################
########### for highway-_env
HIGHWAYENV_OBS_KINEMATICS = _assignment(0)
HIGHWAYENV_OBS_GRAYIMG = _assignment()
HIGHWAYENV_OBS_OCCUPANCY = _assignment()
HIGHWAYENV_OBS_TTC = _assignment()

HIGHWAYENV_ACT_META = _assignment(0)
HIGHWAYENV_ACT_DISCRETE = _assignment()
HIGHWAYENV_ACT_CONTINUOUS = _assignment()

# del _i, _assignment
del _assignment

# if __name__ == '__main__':
#     pass
