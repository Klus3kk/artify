import subprocess
import logging


class DockerHandler:
    @staticmethod
    def build_image(dockerfile_path: str, image_name: str):
        """
        Build a Docker image from the specified Dockerfile.
        :param dockerfile_path: Path to the Dockerfile.
        :param image_name: Name to tag the built image.
        """
        try:
            command = ["docker", "build", "-f", dockerfile_path, "-t", image_name, "."]
            subprocess.run(command, check=True)
            logging.info(f"Docker image '{image_name}' built successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to build Docker image '{image_name}': {e}")
            raise

    @staticmethod
    def run_container(image_name: str, container_name: str, ports: dict, volumes: dict = None):
        """
        Run a Docker container from an image.
        :param image_name: Name of the Docker image.
        :param container_name: Name for the running container.
        :param ports: Port mappings in the form {host_port: container_port}.
        :param volumes: Volume mappings in the form {host_path: container_path}.
        """
        try:
            command = ["docker", "run", "-d", "--name", container_name]
            for host_port, container_port in ports.items():
                command.extend(["-p", f"{host_port}:{container_port}"])
            if volumes:
                for host_path, container_path in volumes.items():
                    command.extend(["-v", f"{host_path}:{container_path}"])
            command.append(image_name)
            subprocess.run(command, check=True)
            logging.info(f"Docker container '{container_name}' started successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to run Docker container '{container_name}': {e}")
            raise

    @staticmethod
    def stop_container(container_name: str):
        """
        Stop a running Docker container.
        :param container_name: Name of the container to stop.
        """
        try:
            command = ["docker", "stop", container_name]
            subprocess.run(command, check=True)
            logging.info(f"Docker container '{container_name}' stopped successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to stop Docker container '{container_name}': {e}")
            raise

    @staticmethod
    def push_image(image_name: str, registry_url: str):
        """
        Push a Docker image to a registry.
        :param image_name: Name of the Docker image.
        :param registry_url: Registry URL to push the image to.
        """
        try:
            tagged_image = f"{registry_url}/{image_name}"
            subprocess.run(["docker", "tag", image_name, tagged_image], check=True)
            subprocess.run(["docker", "push", tagged_image], check=True)
            logging.info(f"Docker image '{image_name}' pushed to '{registry_url}'.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to push Docker image '{image_name}' to '{registry_url}': {e}")
            raise
