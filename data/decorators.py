import spider

def logbuffer_plan(plan_func):
    '''
    此装饰器，将planner的plan函数，包装成logbuffer_plan函数。
    监听plan函数的输入输出，存入log buffer
    '''
    if not hasattr(logbuffer_plan, "t"):
        logbuffer_plan.t = 0.0

    if hasattr(plan_func, "__self__"):
        # 将实例化后的planner的plan函数，封装起来。
        # 用于在planner已经实例化后，在外部加装装饰器（logbuffer.apply_to()）
        planner_instance: spider.planner_zoo.BasePlanner = plan_func.__self__  # 实例化后的对象
        def wrapper(*args, **kwargs):
            plan = plan_func(*args, **kwargs)

            if getattr(planner_instance, "_activate_log_buffer"):
                timestamp = logbuffer_plan.t
                observation = args[:3] # todo:这里有问题，observation其中有的量可能会以kwargs输入，看看怎么处理
                plan = plan
                planner_instance.log_buffer.record_forward(timestamp, observation, plan)

            logbuffer_plan.t += planner_instance.dt
            return plan

    else:
        # 将实例化前的planner的plan函数，封装起来。
        # 用于在planner代码构建的时候，用@logbuffer_plan来装饰plan函数

        def wrapper(*args, **kwargs):
            plan = plan_func(*args, **kwargs)

            planner_instance: spider.planner_zoo.BasePlanner = args[0]
            if getattr(planner_instance, "_activate_log_buffer"):
                timestamp = logbuffer_plan.t
                observation = args[1:4]  # todo:这里有问题，observation其中有的量可能会以kwargs输入，看看怎么处理
                plan = plan
                planner_instance.log_buffer.record_forward(timestamp, observation, plan)

            logbuffer_plan.t += planner_instance.dt # todo:目前这个时间戳的计算方式有问题
            return plan

    return wrapper


def expbuffer_policy(forward_func):
    '''
    此装饰器，将策略网络的forward函数，包装成expbuffer_policy函数。
    '''
    # if not hasattr(expbuffer_policy, "t"):
    #     expbuffer_policy.t = 0.0

    if hasattr(forward_func, "__self__"):
        # 将实例化后的policy的forward函数，封装起来。
        # 用于在policy已经实例化后，在外部加装装饰器（expbuffer.apply_to()）
        policy_instance = forward_func.__self__  # 实例化后的对象
        def wrapper(*args, **kwargs):
            action = forward_func(*args, **kwargs)

            if getattr(policy_instance, "_activate_exp_buffer"):
                timestamp = None#expbuffer_policy.t # 暂时不储存timestamp
                state = args[0] # todo:state一定都会放在第一个吗，可能得统一一下policy的输入输出形式
                action = action
                policy_instance._exp_buffer.record_forward(timestamp, state, action)

            # expbuffer_policy.t += policy_instance.dt if hasattr(policy_instance, "dt") else 1
            return action

    else:
        # 将实例化前的policy的forward函数，封装起来。
        # 用于在policy代码构建的时候，用@expbuffer_policy来装饰forward函数
        def wrapper(*args, **kwargs):
            action = forward_func(*args, **kwargs)
            policy_instance = args[0]

            if getattr(policy_instance, "_activate_exp_buffer"):
                timestamp = None  # expbuffer_policy.t
                state = args[1]  # todo:state一定都会放在第一个吗，可能得统一一下policy的输入输出形式
                action = action
                policy_instance._exp_buffer.record_forward(timestamp, state, action)

            # expbuffer_policy.t += policy_instance.dt if hasattr(policy_instance, "dt") else 1
            return action

    return wrapper