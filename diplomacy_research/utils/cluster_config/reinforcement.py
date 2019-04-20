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
""" Reinforcement Cluster Configuration
    - Retrieves a cluster configuration to be used for reinforcement learning
"""
import logging
import os
from diplomacy_research.utils.cluster import get_nb_gpus, get_cluster_nodes, get_hostname, get_ip_address, \
    OTHER_ROLE_OFFSET, ClusterConfig

# Constants
LOGGER = logging.getLogger(__name__)

def get_cluster_config(gpu_id, nb_param_servers=3, full_config=True, **kwargs):
    """ Retrieves a cluster configuration for the current process
        Note: This function is for reinforcement learning and assumes the following split:
            ps / actor / learner / evaluator / redis / serving

        :param gpu_id: Integer representing the gpu_id of the current process.
                       A gpu_id >= 100 indicates a separate process to launch off the gpu.
        :param nb_param_servers: Optional. Will try to launch this number of parameter servers
        :param full_config: Boolean. If True, loading all the attributes (and might load Tensorflow).
                                     If False, only loads the cluster_spec, job_name, task_id, is_chief, cuda_device_id.
        :return: A ClusterConfig tuple with the cluster configuration
                 Note: A value of None will be returned if the process is not part of the cluster

    Possible kwargs:
        :param nb_cpu_workers: Optional. If launching on CPU, determines the number of CPU workers. (For debugging)
        :param grpc_port: The first port number to open for distributed training
    """
    nb_param_servers = max(1, nb_param_servers)
    nb_cpu_workers = kwargs.get('debug_nb_cpu_workers', kwargs.get('nb_cpu_workers', 0))
    grpc_port = kwargs.get('grpc_port', 2200)

    # Detecting invalid args
    # 1) Negative GPU specified
    # 2) No PS requested
    # 3) GPUs requested (for distributed training), but less than 2 found
    if gpu_id < 0 \
            or nb_param_servers <= 0 \
            or (nb_cpu_workers == 0 and get_nb_gpus() <= 1):
        return None

    # Determining the number of GPUs on the current node
    nb_gpus_per_node = get_nb_gpus()
    using_cpu_only = nb_cpu_workers > 0

    # Determining the nodes on the network
    nodes = get_cluster_nodes()                     # {node_host => node_ip}
    current_node = get_hostname()
    if current_node not in nodes:
        LOGGER.warning('The current node "%s" is not part of the clusters.', current_node)
        LOGGER.warning('Other nodes: %s', [hostname for hostname in nodes])
        return None

    # Building cluster spec
    ps_specs = _get_ps_specs(nodes, nb_param_servers, using_cpu_only, grpc_port)
    actor_specs = _get_actor_specs(nodes, nb_gpus_per_node, using_cpu_only, nb_cpu_workers, grpc_port)
    learner_specs = _get_learner_specs(nodes, nb_gpus_per_node, using_cpu_only, nb_cpu_workers, grpc_port)
    evaluator_specs = _get_evaluator_specs(nodes, grpc_port)
    serving_specs = _get_serving_specs(nodes, nb_gpus_per_node, using_cpu_only, nb_cpu_workers, grpc_port)
    redis_specs = _get_redis_specs(nodes, nb_param_servers)
    cluster_spec = {'ps': ps_specs,
                    'actor': actor_specs,
                    'learner': learner_specs,
                    'evaluator': evaluator_specs,
                    'serving': serving_specs,
                    'redis': redis_specs}

    # Retrieving other information
    job_name, task_id = _get_job_name_task_id(gpu_id, cluster_spec, grpc_port)
    is_chief = bool(job_name == 'learner' and task_id == 0)
    process_pool_cores = _get_process_pool_cores(job_name)
    cuda_device_id = _get_cuda_device_id(gpu_id, job_name, using_cpu_only)

    # Not a valid role
    if job_name not in ['ps', 'actor', 'learner', 'evaluator', 'serving', 'redis']:
        return None

    # Returning minimal config (without Tensorflow)
    if not full_config:
        return ClusterConfig(cluster_spec=cluster_spec,
                             job_name=job_name,
                             task_id=task_id,
                             worker_device=None,
                             iterator_device=None,
                             caching_device=None,
                             partitioner=None,
                             is_chief=is_chief,
                             protocol=None,
                             num_shards=None,
                             shard_index=None,
                             process_pool_cores=process_pool_cores,
                             cuda_device_id=cuda_device_id)

    # Strategies and devices
    ps_strategy = _get_ps_strategy(job_name, ps_specs)
    worker_device = _get_worker_device(job_name, cluster_spec, task_id, ps_strategy)
    iterator_device = _get_iterator_device(job_name, task_id)
    caching_device = _get_caching_device(job_name, task_id)
    protocol = 'grpc'
    num_shards = _get_num_shards(job_name, learner_specs)
    shard_index = _get_shard_index(job_name, task_id)

    # Returning full config
    return ClusterConfig(cluster_spec=cluster_spec,
                         job_name=job_name,
                         task_id=task_id,
                         worker_device=worker_device,
                         iterator_device=iterator_device,
                         caching_device=caching_device,
                         partitioner=None,
                         is_chief=is_chief,
                         protocol=protocol,
                         num_shards=num_shards,
                         shard_index=shard_index,
                         process_pool_cores=process_pool_cores,
                         cuda_device_id=cuda_device_id)

# ---------------------------------------------
# ------       HELPER FUNCTIONS        --------
# ---------------------------------------------
def _get_ps_specs(nodes, nb_param_servers, using_cpu_only, grpc_port):
    """ Returns the list of ip:port for all ps nodes """
    ps_specs = []
    ps_nodes = [hostname for hostname in nodes][:nb_param_servers]

    # Building list
    # Launching 1 per ps_node, except when debugging
    for node_host, node_ip in nodes.items():
        if node_host in ps_nodes:
            if using_cpu_only:
                for ps_ix in range(nb_param_servers):
                    ps_specs += ['%s:%d' % (node_ip, grpc_port + 100 + ps_ix)]
            else:
                ps_specs += ['%s:%d' % (node_ip, grpc_port + 100)]
    return ps_specs

def _get_actor_specs(nodes, nb_gpus_per_node, using_cpu_only, nb_cpu_workers, grpc_port):
    """ Returns the list of ip:port for all actor nodes """
    # There is one actor for each serving
    gpu_ips = []
    for node_ip in nodes.values():
        for gpu_ix in range(nb_cpu_workers if using_cpu_only else nb_gpus_per_node):
            gpu_ips += ['%s:%d' % (node_ip, grpc_port + 200 + gpu_ix)]

    # One actor per serving, 12.5% of GPUs are learners, rest are serving
    nb_learners = max(len(gpu_ips) // 8, 1)
    return gpu_ips[:-nb_learners]

def _get_learner_specs(nodes, nb_gpus_per_node, using_cpu_only, nb_cpu_workers, grpc_port):
    """ Returns the list of ip:port for all learner nodes """
    gpu_ips = []
    for node_ip in nodes.values():
        for gpu_ix in range(nb_cpu_workers if using_cpu_only else nb_gpus_per_node):
            gpu_ips += ['%s:%d' % (node_ip, grpc_port + gpu_ix)]

    # 12.5% of nodes are learners, the rest are serving
    nb_learners = max(len(gpu_ips) // 8, 1)
    return gpu_ips[-nb_learners:]

def _get_evaluator_specs(nodes, grpc_port):
    """ Returns the list of ip:port for all evaluator nodes """
    # There is one evaluator per node
    return ['%s:%d' % (node_ip, grpc_port + 300) for node_ip in nodes.values()]

def _get_serving_specs(nodes, nb_gpus_per_node, using_cpu_only, nb_cpu_workers, grpc_port):
    """ Returns the list of ip:port for all TF serving nodes """
    gpu_ips = []
    for node_ip in nodes.values():
        for gpu_ix in range(nb_cpu_workers if using_cpu_only else nb_gpus_per_node):
            gpu_ips += ['%s:%d' % (node_ip, grpc_port + gpu_ix)]

    # 12.5% of nodes are learners, the rest are serving
    nb_learners = max(len(gpu_ips) // 8, 1)
    return gpu_ips[:-nb_learners]

def _get_redis_specs(nodes, nb_param_servers):
    """ Returns the list of ip:port for all redis nodes """
    redis_specs = []
    redis_nodes = [hostname for hostname in nodes][:nb_param_servers + 1][-1]
    for node_host, node_ip in nodes.items():
        if node_host in redis_nodes:
            redis_specs += ['%s:%d' % (node_ip, 6379)]
    return redis_specs

def _get_job_name_task_id(gpu_id, cluster_spec, grpc_port):
    """ Returns the job name and task id for the current gpu """
    get_spec = lambda task: [spec for spec in cluster_spec.get(task, [])]
    get_nb_tasks = lambda specs, hostname: len([node for node in specs if '%s:' % get_ip_address(hostname) in node])

    job_name, task_id = None, None
    current_node = get_hostname()
    node_spec = '%s:%d' % (get_ip_address(current_node), grpc_port + gpu_id)
    ps_specs = get_spec('ps')
    actor_specs = get_spec('actor')
    learner_specs = get_spec('learner')
    evaluator_specs = get_spec('evaluator')
    serving_specs = get_spec('serving')
    redis_specs = get_spec('redis')

    nb_ps_tasks = get_nb_tasks(ps_specs, current_node)
    nb_redis_tasks = get_nb_tasks(redis_specs, current_node)
    nb_actor_tasks = get_nb_tasks(actor_specs, current_node)
    nb_evaluator_tasks = get_nb_tasks(evaluator_specs, current_node)

    # PS / Actor / Evaluator / Redis        (i.e. not directly using the GPUs)
    if gpu_id >= OTHER_ROLE_OFFSET:
        gpu_id -= OTHER_ROLE_OFFSET

        # PS
        if 0 <= gpu_id < nb_ps_tasks:
            node_spec = '%s:%d' % (get_ip_address(current_node), grpc_port + 100 + gpu_id)
            job_name = 'ps'
            task_id = ps_specs.index(node_spec)
        gpu_id -= nb_ps_tasks

        # Actor
        if 0 <= gpu_id < nb_actor_tasks:
            node_spec = '%s:%d' % (get_ip_address(current_node), grpc_port + 200 + gpu_id)
            job_name = 'actor'
            task_id = actor_specs.index(node_spec)
        gpu_id -= nb_actor_tasks

        # Redis
        if nb_redis_tasks == 1 and gpu_id == 0:
            job_name = 'redis'
            task_id = 0
        gpu_id -= nb_redis_tasks

        # Evaluator
        if nb_evaluator_tasks == 1 and gpu_id == 0:
            node_spec = '%s:%d' % (get_ip_address(current_node), grpc_port + 300)
            job_name = 'evaluator'
            task_id = evaluator_specs.index(node_spec)
        gpu_id -= nb_evaluator_tasks

    # Serving
    elif node_spec in serving_specs:
        job_name = 'serving'
        task_id = serving_specs.index(node_spec)

    # Learner
    elif node_spec in learner_specs:
        job_name = 'learner'
        task_id = learner_specs.index(node_spec)

    # Returning job_name, task_id
    return job_name, task_id

def _get_ps_strategy(job_name, ps_specs):
    """ Returns the param server strategy to use """
    if job_name != 'learner' or len(ps_specs) <= 1:
        return None
    from diplomacy_research.utils.tensorflow import GreedyLoadBalancingStrategy, byte_size_load_fn
    return GreedyLoadBalancingStrategy(num_tasks=len(ps_specs), load_fn=byte_size_load_fn)

def _get_worker_device(job_name, cluster_spec, task_id, ps_strategy):
    """ Returns the worker device to use """
    if job_name != 'learner':
        return None
    from diplomacy_research.utils.tensorflow import tf
    return tf.train.replica_device_setter(cluster=tf.train.ClusterSpec(cluster_spec),
                                          worker_device='/job:%s/task:%d' % (job_name, task_id),
                                          ps_strategy=ps_strategy)

def _get_iterator_device(job_name, task_id):
    """ Returns the iterator device to use """
    if job_name != 'learner':
        return None
    return '/job:%s/task:%d' % (job_name, task_id)

def _get_caching_device(job_name, task_id):
    """ Returns the local caching device for a given job name / task id """
    if job_name != 'learner':
        return None
    return '/job:%s/task:%d' % (job_name, task_id)

def _get_num_shards(job_name, learner_specs):
    """ Returns num of shards on the cluster """
    if job_name != 'learner':
        return None
    return len(learner_specs)

def _get_shard_index(job_name, task_id):
    """ Returns the current shard index """
    if job_name != 'learner':
        return None
    return task_id

def _get_process_pool_cores(job_name):
    """ Returns the number of cores to use for the process pool """
    if job_name in ('actor', 'evaluator'):
        return os.cpu_count()
    return 0

def _get_cuda_device_id(gpu_id, job_name, using_cpu_only):
    """ Returns the CUDA device id """
    if using_cpu_only or job_name not in ('serving', 'learner'):
        return ''
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if len(cuda_visible_devices.split(',')) > gpu_id:
        return cuda_visible_devices.split(',')[gpu_id]
    return cuda_visible_devices
