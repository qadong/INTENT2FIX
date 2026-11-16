from typing import Optional, List, Type, ClassVar

from pydantic import Field, model_validator, ConfigDict

from INTENT2FIX.actions.schema import ActionArguments, FewShotExample

from INTENT2FIX.actions.docker_base import DockerEnv, DockerEnvArgs, DockerEnvObservation

from INTENT2FIX.actions.search_base import SearchBaseAction, SearchBaseArgs
from INTENT2FIX.index.types import SearchCodeResponse

import os 
import time
from docker import DockerClient
from docker.errors import DockerException
from typing import Optional, Dict, Any

class ReproduceArgs(DockerEnvArgs):
    reproduce_script: str = Field(
        default="arvo",
        description="The script or command to run for the reproduce process.",
    )
    input_files: Optional[List[str]] = Field(
        default=None,
        description="List of input files required for the reproduce process.",
    )
    output_dir: Optional[str] = Field(
        default="/output",
        description="The directory inside the container where output files will be saved.",
    )
    timeout: Optional[int] = Field(
        default=600,
        description="Maximum time (in seconds) allowed for the reproduce process to complete.",
    )



class Reproduce(DockerEnv):
    args_schema: ClassVar[Type[ActionArguments]] = ReproduceArgs

    def execute(
        self,
        args: ReproduceArgs,
        file_context: str | None = None,
        workspace: str | None = None,
    ) -> DockerEnvObservation:
        try:
            # 设置 reproduce 任务的命令
            args.command = args.reproduce_script

            # 挂载整个 workspace（如果提供了）
            if workspace:
                volumes = args.volumes or {}
                volumes[workspace.path] = {"bind": "/workspace", "mode": "rw"}
                args.volumes = volumes

            # 挂载输入文件（如果提供了）
            if args.input_files and workspace:
                for input_file in args.input_files:
                    host_path = os.path.join(workspace.path, input_file)
                    container_path = os.path.join("/input", input_file)
                    volumes[host_path] = {"bind": container_path, "mode": "ro"}

            # 挂载输出目录
            if workspace:
                volumes[workspace.output_dir] = {"bind": args.output_dir, "mode": "rw"}

            # 设置超时时间
            start_time = time.time()

            # 调用基类的 _run_container 方法运行容器
            observation = self._run_container(args)

            # 检查是否超时
            elapsed_time = time.time() - start_time
            if elapsed_time > args.timeout:
                observation.message = "Reproduce process timed out."
                observation.summary = f"Reproduce process exceeded the timeout of {args.timeout} seconds."
                observation.properties["timeout"] = True
                observation.properties["elapsed_time"] = elapsed_time

            # 分析日志
            if observation.logs:
                observation.properties["log_analysis"] = self._analyze_logs(observation.logs)

            # 保存输出到本地文件（如果提供了 workspace）
            if workspace and observation.logs:
                output_file = os.path.join(workspace.output_dir, "arvo_output.log")
                with open(output_file, "w") as f:
                    f.write(observation.logs)
                observation.properties["output_file"] = output_file

            return observation

        except DockerException as e:
            return DockerEnvObservation(
                message=f"Failed to execute reproduce process: {str(e)}",
                summary="Reproduce process failed due to Docker error.",
                properties={"error": str(e)},
            )
            
    def run_reproduce(
        self,
        docker_image: str,
        reproduce_script: str = "arvo",
        input_files: Optional[List[str]] = None,
        output_dir: str = "/output",
        timeout: int = 300,
        workspace: Optional[str] = None,
    ) -> DockerEnvObservation:
        """
        Run the reproduce process using the given arguments.
        """
        # 设置参数
        args = ReproduceArgs(
            docker_image=docker_image,
            reproduce_script=reproduce_script,
            input_files=input_files,
            output_dir=output_dir,
            timeout=timeout,
        )

        # 执行复现任务
        return self.execute(args, workspace=workspace)         
     

    def _analyze_logs(self, logs: str) -> Dict[str, Any]:
        """
        analyze the logs and return a dictionary of relevant metrics
        """
        log_analysis = {
            "error_count": logs.lower().count("error"),
            "warning_count": logs.lower().count("warning"),
            "success": "success" in logs.lower(),
        }
        return log_analysis
    
if __name__ == "__main__":
        
        
    # reproduce = Reproduce(docker_client=DockerClient.from_env())

    # # 设置参数
    # args = ReproduceArgs(
    #     docker_image="docker.tbedu.top/n132/arvo:25402-vul",
    #     reproduce_script="arvo",  # 执行 arvo 命令
    #     input_files=None,
    #     output_dir="/output",
    #     timeout=300,
    # )

    # # 执行复现任务
    # observation = reproduce.execute(args, workspace=None)

    # # 输出结果
    # print('observation.message','_______________________________')
    # print(observation.message)
    # print('observation.logs','______________________________')
    # print(observation.logs)
    # print('observation.properties','__________________________')
    # print(observation.properties)
    
    
    reproduce = Reproduce(docker_client=DockerClient.from_env())

    observation = reproduce.run_reproduce(
        docker_image="docker.tbedu.top/n132/arvo:25402-vul",
        reproduce_script="arvo",
        input_files=None,
        output_dir="/output",
        timeout=300,
        workspace=None,
    )

    # 输出结果
    print(observation.message)
    print(observation.logs)
    print(observation.properties)