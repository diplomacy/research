# ==============================================================================
# Copyright 2019 - Philip Paquette
#
# NOTICE:  Permission is hereby granted, free of charge, to any person obtaining
#   a copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
# ==============================================================================
""" Tests for Reinforcement get_cluster_config """
import os
from diplomacy_research.utils.cluster_config.reinforcement import get_cluster_config


def test_3_ps_3_worker():
    """ Tests with 3 param servers and 3 workers """
    # pylint: disable=too-many-statements
    kwargs = {'nb_param_servers': 3, 'nb_cpu_workers': 3}

    # GPU id of 0
    cluster_config = get_cluster_config(gpu_id=0, **kwargs)
    cluster_spec = cluster_config.cluster_spec
    assert sorted(list(cluster_spec.keys())) == sorted(['ps', 'actor', 'learner', 'evaluator', 'serving', 'redis'])
    assert cluster_spec['ps'] == ['localhost:2300', 'localhost:2301', 'localhost:2302']
    assert cluster_spec['learner'] == ['localhost:2202']
    assert cluster_spec['actor'] == ['localhost:2400', 'localhost:2401']
    assert cluster_spec['evaluator'] == ['localhost:2500']
    assert cluster_spec['serving'] == ['localhost:2200', 'localhost:2201']
    assert cluster_spec['redis'] == ['localhost:6379']
    assert cluster_config.job_name == 'serving'
    assert cluster_config.task_id == 0
    assert cluster_config.worker_device is None
    assert cluster_config.iterator_device is None
    assert cluster_config.caching_device is None
    assert cluster_config.partitioner is None
    assert not cluster_config.is_chief
    assert cluster_config.protocol == 'grpc'
    assert cluster_config.num_shards is None
    assert cluster_config.shard_index is None
    assert cluster_config.process_pool_cores == 0
    assert cluster_config.cuda_device_id == ''

    # GPU id of 1
    cluster_config = get_cluster_config(gpu_id=1, **kwargs)
    cluster_spec = cluster_config.cluster_spec
    assert sorted(list(cluster_spec.keys())) == sorted(['ps', 'actor', 'learner', 'evaluator', 'serving', 'redis'])
    assert cluster_spec['ps'] == ['localhost:2300', 'localhost:2301', 'localhost:2302']
    assert cluster_spec['learner'] == ['localhost:2202']
    assert cluster_spec['actor'] == ['localhost:2400', 'localhost:2401']
    assert cluster_spec['evaluator'] == ['localhost:2500']
    assert cluster_spec['serving'] == ['localhost:2200', 'localhost:2201']
    assert cluster_spec['redis'] == ['localhost:6379']
    assert cluster_config.job_name == 'serving'
    assert cluster_config.task_id == 1
    assert cluster_config.worker_device is None
    assert cluster_config.iterator_device is None
    assert cluster_config.caching_device is None
    assert cluster_config.partitioner is None
    assert not cluster_config.is_chief
    assert cluster_config.protocol == 'grpc'
    assert cluster_config.num_shards is None
    assert cluster_config.shard_index is None
    assert cluster_config.process_pool_cores == 0
    assert cluster_config.cuda_device_id == ''

    # GPU id of 2
    cluster_config = get_cluster_config(gpu_id=2, **kwargs)
    cluster_spec = cluster_config.cluster_spec
    assert sorted(list(cluster_spec.keys())) == sorted(['ps', 'actor', 'learner', 'evaluator', 'serving', 'redis'])
    assert cluster_spec['ps'] == ['localhost:2300', 'localhost:2301', 'localhost:2302']
    assert cluster_spec['learner'] == ['localhost:2202']
    assert cluster_spec['actor'] == ['localhost:2400', 'localhost:2401']
    assert cluster_spec['evaluator'] == ['localhost:2500']
    assert cluster_spec['serving'] == ['localhost:2200', 'localhost:2201']
    assert cluster_spec['redis'] == ['localhost:6379']
    assert cluster_config.job_name == 'learner'
    assert cluster_config.task_id == 0
    assert callable(cluster_config.worker_device)
    assert cluster_config.iterator_device == '/job:learner/task:0'
    assert cluster_config.caching_device == '/job:learner/task:0'
    assert cluster_config.partitioner is None
    assert cluster_config.is_chief
    assert cluster_config.protocol == 'grpc'
    assert cluster_config.num_shards == 1
    assert cluster_config.shard_index == 0
    assert cluster_config.process_pool_cores == 0
    assert cluster_config.cuda_device_id == ''

    # GPU id of 3
    assert get_cluster_config(gpu_id=3, **kwargs) is None

    # GPU id of 100 (ps 0)
    cluster_config = get_cluster_config(gpu_id=100, **kwargs)
    cluster_spec = cluster_config.cluster_spec
    assert sorted(list(cluster_spec.keys())) == sorted(['ps', 'actor', 'learner', 'evaluator', 'serving', 'redis'])
    assert cluster_spec['ps'] == ['localhost:2300', 'localhost:2301', 'localhost:2302']
    assert cluster_spec['learner'] == ['localhost:2202']
    assert cluster_spec['actor'] == ['localhost:2400', 'localhost:2401']
    assert cluster_spec['evaluator'] == ['localhost:2500']
    assert cluster_spec['serving'] == ['localhost:2200', 'localhost:2201']
    assert cluster_spec['redis'] == ['localhost:6379']
    assert cluster_config.job_name == 'ps'
    assert cluster_config.task_id == 0
    assert cluster_config.worker_device is None
    assert cluster_config.iterator_device is None
    assert cluster_config.caching_device is None
    assert cluster_config.partitioner is None
    assert not cluster_config.is_chief
    assert cluster_config.protocol == 'grpc'
    assert cluster_config.num_shards is None
    assert cluster_config.shard_index is None
    assert cluster_config.process_pool_cores == 0
    assert cluster_config.cuda_device_id == ''

    # GPU id of 101 (ps 1)
    cluster_config = get_cluster_config(gpu_id=101, **kwargs)
    cluster_spec = cluster_config.cluster_spec
    assert sorted(list(cluster_spec.keys())) == sorted(['ps', 'actor', 'learner', 'evaluator', 'serving', 'redis'])
    assert cluster_spec['ps'] == ['localhost:2300', 'localhost:2301', 'localhost:2302']
    assert cluster_spec['learner'] == ['localhost:2202']
    assert cluster_spec['actor'] == ['localhost:2400', 'localhost:2401']
    assert cluster_spec['evaluator'] == ['localhost:2500']
    assert cluster_spec['serving'] == ['localhost:2200', 'localhost:2201']
    assert cluster_spec['redis'] == ['localhost:6379']
    assert cluster_config.job_name == 'ps'
    assert cluster_config.task_id == 1
    assert cluster_config.worker_device is None
    assert cluster_config.iterator_device is None
    assert cluster_config.caching_device is None
    assert cluster_config.partitioner is None
    assert not cluster_config.is_chief
    assert cluster_config.protocol == 'grpc'
    assert cluster_config.num_shards is None
    assert cluster_config.shard_index is None
    assert cluster_config.process_pool_cores == 0
    assert cluster_config.cuda_device_id == ''

    # GPU id of 102 (ps 2)
    cluster_config = get_cluster_config(gpu_id=102, **kwargs)
    cluster_spec = cluster_config.cluster_spec
    assert sorted(list(cluster_spec.keys())) == sorted(['ps', 'actor', 'learner', 'evaluator', 'serving', 'redis'])
    assert cluster_spec['ps'] == ['localhost:2300', 'localhost:2301', 'localhost:2302']
    assert cluster_spec['learner'] == ['localhost:2202']
    assert cluster_spec['actor'] == ['localhost:2400', 'localhost:2401']
    assert cluster_spec['evaluator'] == ['localhost:2500']
    assert cluster_spec['serving'] == ['localhost:2200', 'localhost:2201']
    assert cluster_spec['redis'] == ['localhost:6379']
    assert cluster_config.job_name == 'ps'
    assert cluster_config.task_id == 2
    assert cluster_config.worker_device is None
    assert cluster_config.iterator_device is None
    assert cluster_config.caching_device is None
    assert cluster_config.partitioner is None
    assert not cluster_config.is_chief
    assert cluster_config.protocol == 'grpc'
    assert cluster_config.num_shards is None
    assert cluster_config.shard_index is None
    assert cluster_config.process_pool_cores == 0
    assert cluster_config.cuda_device_id == ''

    # GPU id of 103 (Actor 0)
    cluster_config = get_cluster_config(gpu_id=103, **kwargs)
    cluster_spec = cluster_config.cluster_spec
    assert sorted(list(cluster_spec.keys())) == sorted(['ps', 'actor', 'learner', 'evaluator', 'serving', 'redis'])
    assert cluster_spec['ps'] == ['localhost:2300', 'localhost:2301', 'localhost:2302']
    assert cluster_spec['learner'] == ['localhost:2202']
    assert cluster_spec['actor'] == ['localhost:2400', 'localhost:2401']
    assert cluster_spec['evaluator'] == ['localhost:2500']
    assert cluster_spec['serving'] == ['localhost:2200', 'localhost:2201']
    assert cluster_spec['redis'] == ['localhost:6379']
    assert cluster_config.job_name == 'actor'
    assert cluster_config.task_id == 0
    assert cluster_config.worker_device is None
    assert cluster_config.iterator_device is None
    assert cluster_config.caching_device is None
    assert cluster_config.partitioner is None
    assert not cluster_config.is_chief
    assert cluster_config.protocol == 'grpc'
    assert cluster_config.num_shards is None
    assert cluster_config.shard_index is None
    assert cluster_config.process_pool_cores == os.cpu_count()
    assert cluster_config.cuda_device_id == ''

    # GPU id of 104 (Actor 1)
    cluster_config = get_cluster_config(gpu_id=104, **kwargs)
    cluster_spec = cluster_config.cluster_spec
    assert sorted(list(cluster_spec.keys())) == sorted(['ps', 'actor', 'learner', 'evaluator', 'serving', 'redis'])
    assert cluster_spec['ps'] == ['localhost:2300', 'localhost:2301', 'localhost:2302']
    assert cluster_spec['learner'] == ['localhost:2202']
    assert cluster_spec['actor'] == ['localhost:2400', 'localhost:2401']
    assert cluster_spec['evaluator'] == ['localhost:2500']
    assert cluster_spec['serving'] == ['localhost:2200', 'localhost:2201']
    assert cluster_spec['redis'] == ['localhost:6379']
    assert cluster_config.job_name == 'actor'
    assert cluster_config.task_id == 1
    assert cluster_config.worker_device is None
    assert cluster_config.iterator_device is None
    assert cluster_config.caching_device is None
    assert cluster_config.partitioner is None
    assert not cluster_config.is_chief
    assert cluster_config.protocol == 'grpc'
    assert cluster_config.num_shards is None
    assert cluster_config.shard_index is None
    assert cluster_config.process_pool_cores == os.cpu_count()
    assert cluster_config.cuda_device_id == ''

    # GPU id of 105 (Redis 0)
    cluster_config = get_cluster_config(gpu_id=105, **kwargs)
    cluster_spec = cluster_config.cluster_spec
    assert sorted(list(cluster_spec.keys())) == sorted(['ps', 'actor', 'learner', 'evaluator', 'serving', 'redis'])
    assert cluster_spec['ps'] == ['localhost:2300', 'localhost:2301', 'localhost:2302']
    assert cluster_spec['learner'] == ['localhost:2202']
    assert cluster_spec['actor'] == ['localhost:2400', 'localhost:2401']
    assert cluster_spec['evaluator'] == ['localhost:2500']
    assert cluster_spec['serving'] == ['localhost:2200', 'localhost:2201']
    assert cluster_spec['redis'] == ['localhost:6379']
    assert cluster_config.job_name == 'redis'
    assert cluster_config.task_id == 0
    assert cluster_config.worker_device is None
    assert cluster_config.iterator_device is None
    assert cluster_config.caching_device is None
    assert cluster_config.partitioner is None
    assert not cluster_config.is_chief
    assert cluster_config.protocol == 'grpc'
    assert cluster_config.num_shards is None
    assert cluster_config.shard_index is None
    assert cluster_config.process_pool_cores == 0
    assert cluster_config.cuda_device_id == ''

    # GPU id of 106 (Evaluator 0)
    cluster_config = get_cluster_config(gpu_id=106, **kwargs)
    cluster_spec = cluster_config.cluster_spec
    assert sorted(list(cluster_spec.keys())) == sorted(['ps', 'actor', 'learner', 'evaluator', 'serving', 'redis'])
    assert cluster_spec['ps'] == ['localhost:2300', 'localhost:2301', 'localhost:2302']
    assert cluster_spec['learner'] == ['localhost:2202']
    assert cluster_spec['actor'] == ['localhost:2400', 'localhost:2401']
    assert cluster_spec['evaluator'] == ['localhost:2500']
    assert cluster_spec['serving'] == ['localhost:2200', 'localhost:2201']
    assert cluster_spec['redis'] == ['localhost:6379']
    assert cluster_config.job_name == 'evaluator'
    assert cluster_config.task_id == 0
    assert cluster_config.worker_device is None
    assert cluster_config.iterator_device is None
    assert cluster_config.caching_device is None
    assert cluster_config.partitioner is None
    assert not cluster_config.is_chief
    assert cluster_config.protocol == 'grpc'
    assert cluster_config.num_shards is None
    assert cluster_config.shard_index is None
    assert cluster_config.process_pool_cores == os.cpu_count()
    assert cluster_config.cuda_device_id == ''

    # GPU id of 107 - empty
    assert get_cluster_config(gpu_id=107, **kwargs) is None
