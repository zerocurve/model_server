#
# Copyright (c) 2019-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import functools

from docker.client import DockerClient


def ovms_docker_run(docker_client: DockerClient, target_device :str, *args, **kwargs):
    """
    This function returns a wrapper for docker_client.containers.run with settings adjusted 
    according to passed target_device.
    """
    @functools.wraps(docker_client.containers.run)
    def run(*args, **kwargs):
        if 'MYRIAD' in target_device:
            kwargs['network_mode'] = kwargs.get('network_mode', 'host')
            kwargs['privileged'] = kwargs.get('privileged', True)
            # TODO: handle volumes passed as list?
            kwargs['volumes']['/dev'] = kwargs.get('volumes', {}).get('/dev',
                                                                      {'bind': '/dev',
                                                                       'mode': 'rw'})
        if 'HDDL' in target_device:
            kwargs['devices'] = kwargs.get('devices', ['/dev/ion:/dev/ion'])
            kwargs['volumes']['/var/tmp'] = kwargs.get('volumes', {}).get('/var/tmp',
                                                       {'bind': '/var/tmp', 'mode': 'rw'})
        return docker_client.containers.run(*args, **kwargs)

    return run(*args, **kwargs)