# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MLRun Marketplace Agent Deployment

This module provides functionality to import and deploy agents from the MLRun marketplace.
Agents are code artifacts (e.g., Atomic Agents, LangChain agents) that communicate using
specific protocols (e.g., A2A) and can be deployed as MLRun application runtimes.
"""

import pathlib
from typing import Any, Optional

import yaml

import mlrun
import mlrun.errors
from mlrun.utils import logger


class MarketplaceBackend:
    """
    Marketplace Backend for loading agent configurations.

    Currently loads from YAML files on disk. In production, this will
    connect to the marketplace assets database.
    """

    # Class variable to store custom YAML directory
    _yaml_directory: Optional[str] = None

    @classmethod
    def set_yaml_directory(cls, directory: str):
        """
        Set custom directory for loading agent YAML files (for testing).

        :param directory: Path to directory containing agent YAML files
        """
        cls._yaml_directory = directory

    @staticmethod
    def get(name: str, project: Optional[str] = None) -> "AgentAsset":
        """
        Retrieve agent asset from marketplace.

        Currently loads from YAML files. Supports:
        - "marketplace://agent-name:version" format
        - Direct agent name (uses default version)

        :param name: Agent name (e.g., "marketplace://atomic-agent:0.0.1" or "atomic-agent")
        :param project: Optional project context
        :return: AgentAsset instance
        """
        # Parse agent name
        if name.startswith("marketplace://"):
            agent_name = name.replace("marketplace://", "").split(":")[0]
        else:
            agent_name = name

        # Determine YAML directory
        if MarketplaceBackend._yaml_directory:
            yaml_dir = pathlib.Path(MarketplaceBackend._yaml_directory)
        else:
            # Default: look in mlrun/marketplace directory
            yaml_dir = pathlib.Path(__file__).parent

        # Load YAML file
        yaml_path = yaml_dir / f"{agent_name}.yaml"
        if not yaml_path.exists():
            raise mlrun.errors.MLRunNotFoundError(
                f"Agent '{agent_name}' not found. "
                f"Expected YAML file at: {yaml_path}"
            )

        logger.info("Loading agent from YAML", agent=agent_name, path=str(yaml_path))

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        # Extract configuration
        build_config = config["consumption_config"]["build"]
        deploy_config = config["consumption_config"]["deploy"]

        # Create AgentAsset
        return AgentAsset(
            name=config["name"],
            version=config["version"],
            author=config["author"],
            description=config["description"],
            kind=config["kind"],
            protocol=config["agent_info"]["protocol"],
            framework=config["agent_info"]["framework"],
            asset_url="",  # Will be provided during deployment
            requirements=build_config.get("requirements", []),
            default_base_image=build_config.get("default_base_image", "mlrun/mlrun"),
            default_port=deploy_config.get("default_port", 8080),
            default_command=deploy_config.get("default_command", ""),
            default_args=deploy_config.get("default_args", []),
            default_direct_pod=deploy_config.get("default_direct_pod", False),
            inputs=config["consumption_config"].get("inputs", []),
            categories=config.get("categories", []),
            default_workdir=build_config.get("default_workdir"),
            build_extra=build_config.get("build_extra"),
        )


class AgentAsset:
    """
    Represents an agent asset retrieved from the marketplace.

    Contains all the metadata and configuration needed to deploy the agent,
    including build requirements, deployment defaults, and input specifications.
    """

    def __init__(
        self,
        name: str,
        version: str,
        author: str,
        description: str,
        kind: str,
        protocol: str,
        framework: str,
        asset_url: str,
        requirements: list[str],
        default_base_image: str,
        default_port: int,
        default_command: str,
        default_args: list[str],
        default_direct_pod: bool,
        inputs: list[dict[str, Any]],
        categories: Optional[list[str]] = None,
        default_workdir: Optional[str] = None,
        build_extra: Optional[str] = None,
    ):
        """
        Initialize AgentAsset with marketplace metadata.

        :param name: Agent name
        :param version: Agent version
        :param author: Agent author
        :param description: Agent description
        :param kind: Agent kind (e.g., "atomic-agent")
        :param protocol: Communication protocol (e.g., "A2A")
        :param framework: Implementation framework (e.g., "LangChain")
        :param asset_url: URL to agent code ZIP/archive
        :param requirements: Python requirements list
        :param default_base_image: Default Docker base image
        :param default_port: Default application port
        :param default_command: Default command to run
        :param default_args: Default command arguments
        :param default_direct_pod: Default direct pod access setting
        :param inputs: List of input configurations (secrets/env vars)
        :param categories: Optional list of categories
        :param default_workdir: Optional default working directory for build
        :param build_extra: Optional raw Dockerfile commands for build
        """
        self.name = name
        self.version = version
        self.author = author
        self.description = description
        self.kind = kind
        self.protocol = protocol
        self.framework = framework
        self.asset_url = asset_url
        self.requirements = requirements
        self.default_base_image = default_base_image
        self.default_port = default_port
        self.default_command = default_command
        self.default_args = default_args
        self.default_direct_pod = default_direct_pod
        self.inputs = inputs
        self.categories = categories or []
        self.default_workdir = default_workdir
        self.build_extra = build_extra

        # Extract mandatory configurations from inputs
        # Mandatory = required:true AND no default value
        self.mandatory_configurations = [
            inp["name"]
            for inp in self.inputs
            if inp.get("required", False) and not inp.get("default")
        ]


class MarketplaceAgentDeployer:
    """
    Agent deployer for marketplace agents.

    This class provides methods to get information about an agent and deploy it
    as an MLRun application runtime. It automatically optimizes the build process
    by caching base images with requirements and reusing them across deployments.
    """

    def __init__(self, agent_asset: AgentAsset):
        """
        Initialize MarketplaceAgentDeployer with agent asset metadata.

        :param agent_asset: AgentAsset instance from marketplace
        """
        self.name = agent_asset.name
        self.version = agent_asset.version
        self.description = agent_asset.description
        self.protocol = agent_asset.protocol
        self.framework = agent_asset.framework
        self.author = agent_asset.author
        self.kind = agent_asset.kind
        self.inputs_keys = [inp["name"] for inp in agent_asset.inputs]
        self._agent_asset = agent_asset

        # Track built images for optimization
        self._built_base_image = None  # Image with requirements only
        self._built_final_image = None  # Image with requirements + source

    def info(self) -> str:
        """
        Get information about the agent.

        :return: Formatted string with agent information
        """
        optional_inputs = [
            k
            for k in self.inputs_keys
            if k not in self._agent_asset.mandatory_configurations
        ]

        info_str = (
            f"Agent: {self.name}\n"
            f"Version: {self.version}\n"
            f"Author: {self.author}\n"
            f"Kind: {self.kind}\n"
            f"Description: {self.description}\n"
            f"Framework: {self.framework}\n"
            f"Protocol: {self.protocol}\n"
            f"Required Inputs: "
            f"{', '.join(self._agent_asset.mandatory_configurations)}\n"
            f"Optional Inputs: {', '.join(optional_inputs)}"
        )
        print(info_str)
        return info_str

    def _validate_mandatory_configs(
        self, kwargs: dict[str, Any], mandatory_configs: list[str]
    ) -> dict[str, Any]:
        """
        Validate that all mandatory configurations are provided and apply defaults.

        Mandatory configs (required=true, no default) must be provided by user.
        Optional configs (required=false or has default) use provided value or default.

        :param kwargs: User-provided configurations
        :param mandatory_configs: List of mandatory configuration keys
        :return: Dictionary of validated configurations with defaults applied
        :raises MLRunInvalidArgumentError: If mandatory configs are missing
        """
        configs = {}
        missing_configs = []

        # Check mandatory configs (must be provided)
        for config_key in mandatory_configs:
            if config_key not in kwargs:
                missing_configs.append(config_key)
            else:
                configs[config_key] = kwargs[config_key]

        if missing_configs:
            raise mlrun.errors.MLRunInvalidArgumentError(
                f"Missing mandatory configurations: {', '.join(missing_configs)}"
            )

        # Process all inputs (apply defaults for optional ones)
        for inp in self._agent_asset.inputs:
            key = inp["name"]
            if key in kwargs:
                # User provided value
                configs[key] = kwargs[key]
            elif key not in configs and inp.get("default"):
                # Use default value if not provided and default exists
                configs[key] = inp["default"]

        return configs

    def _get_build_extra_commands(self) -> Optional[str]:
        """
        Generate build extra Dockerfile commands from agent asset configuration.

        Combines default_workdir (if set) and build_extra (if set) into a single
        Dockerfile commands string.

        :return: Dockerfile commands string or None if no extra commands
        """
        commands = []

        # Add WORKDIR if specified
        if self._agent_asset.default_workdir:
            commands.append(f"WORKDIR {self._agent_asset.default_workdir}")

        # Add any additional build extra commands
        if self._agent_asset.build_extra:
            commands.append(self._agent_asset.build_extra.rstrip())

        return "\n".join(commands) + "\n" if commands else None

    def _build_base_image(
        self,
        project_obj: mlrun.projects.MlrunProject,
        base_image: Optional[str] = None,
        requirements: Optional[Any] = None,
    ) -> str:
        """
        Build base image with requirements only (internal method).

        This is an optimization step to avoid rebuilding requirements on every deployment.
        The built image is cached in self._built_base_image for reuse.

        :param project_obj: MLRun project object
        :param base_image: Optional base image to use (defaults to agent's default)
        :param requirements: Optional requirements list or file path (overrides default)
        :return: Built base image URI
        """
        # Determine requirements to use
        reqs = (
            requirements if requirements is not None else self._agent_asset.requirements
        )
        reqs_count = len(reqs) if isinstance(reqs, list) else "from file"

        logger.info(
            "Building base image with requirements",
            agent=self.name,
            requirements_count=reqs_count,
        )

        temp_func_name = f"{self.name}-base-temp"

        # Create temporary function for building base image
        base_func = project_obj.set_function(
            kind="application",
            image=base_image or self._agent_asset.default_base_image,
            name=temp_func_name,
        )

        # Add requirements
        if reqs:
            if isinstance(reqs, str):
                # File path
                base_func.with_requirements(requirements_file=reqs)
            else:
                # List of requirements
                base_func.with_requirements(requirements=reqs)

        # Add build extra commands (WORKDIR, etc.)
        build_extra_commands = self._get_build_extra_commands()
        if build_extra_commands:
            base_func.spec.build.extra = build_extra_commands

        # Build the image
        base_func.build()

        # Get the built image URI
        built_image = base_func.spec.image
        self._built_base_image = built_image

        # Clean up temporary function from project
        try:
            project_obj.delete_function(temp_func_name)
            logger.debug(
                "Cleaned up temporary base build function",
                agent=self.name,
                temp_function=temp_func_name,
            )
        except Exception as exc:
            logger.warning(
                "Failed to clean up temporary function",
                agent=self.name,
                temp_function=temp_func_name,
                error=mlrun.errors.err_to_str(exc),
            )

        logger.info(
            "Base image built successfully",
            agent=self.name,
            image=built_image,
        )

        return built_image

    def _build_with_source(
        self,
        project_obj: mlrun.projects.MlrunProject,
        base_image_with_requirements: str,
    ) -> str:
        """
        Build image with source archive on top of base image (internal method).

        This adds the agent source code to the pre-built base image with requirements.
        The built image is cached in self._built_final_image for reuse.

        :param project_obj: MLRun project object
        :param base_image_with_requirements: Base image with requirements pre-installed
        :return: Built final image URI
        """
        logger.info(
            "Building image with source archive",
            agent=self.name,
            base_image=base_image_with_requirements,
        )

        temp_func_name = f"{self.name}-source-temp"

        # Create function with base image and source
        source_func = project_obj.set_function(
            kind="application",
            image=base_image_with_requirements,
            name=temp_func_name,
        )

        # Add source archive
        source_func.with_source_archive(
            source=self._agent_asset.asset_url, pull_at_runtime=False
        )

        # Add build extra commands (WORKDIR, etc.)
        build_extra_commands = self._get_build_extra_commands()
        if build_extra_commands:
            source_func.spec.build.extra = build_extra_commands

        # Build the image
        source_func.build()

        # Get the built image URI
        built_image = source_func.spec.image
        self._built_final_image = built_image

        # Clean up temporary function from project
        try:
            project_obj.delete_function(temp_func_name)
            logger.debug(
                "Cleaned up temporary source build function",
                agent=self.name,
                temp_function=temp_func_name,
            )
        except Exception as exc:
            logger.warning(
                "Failed to clean up temporary function",
                agent=self.name,
                temp_function=temp_func_name,
                error=mlrun.errors.err_to_str(exc),
            )

        logger.info(
            "Image with source built successfully",
            agent=self.name,
            image=built_image,
        )

        return built_image

    def deploy(
        self,
        project: str,
        source: Optional[str] = None,
        base_image_with_requirements: Optional[str] = None,
        gateway_config: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Deploy the agent as an MLRun application runtime.

        This method automatically optimizes the build process:
        1. If base_image_with_requirements is provided, uses it directly
        2. Otherwise, checks if a base image was already built and reuses it
        3. If no cached images exist, builds from scratch (base + source)

        This transparent optimization significantly speeds up redeployments when
        only configurations change.

        :param project: MLRun project name
        :param source: Source archive URL/path (required if not set in agent asset)
        :param base_image_with_requirements: Pre-built image with requirements
            (skips requirement installation if provided)
        :param gateway_config: API gateway configuration dict. If provided,
            creates an API gateway with these settings. Supports:
            - name: Gateway name (default: "{agent_name}-gateway")
            - path: URL path (default: "/")
            - authentication_mode: Auth mode (e.g., "none", "accessKey")
            - authentication_creds: Auth credentials
            - direct_port_access: Enable direct port access (default: False)
            - ssl_redirect: Enable SSL redirect (default: True)
            - set_as_default: Set as default gateway (default: False)
        :param kwargs: Additional configuration options including:
            - base_image: Override default base image (for initial build)
            - port: Override default port
            - command: Override default command
            - args: Override default args
            - requirements: Override default requirements list or file path
            - direct_pod: Override default direct pod access
            - create_default_api_gateway: Whether to create default API gateway
                (default: False, ignored if gateway_config is provided)
            - Any input configurations (secrets/env vars) as specified in
                agent's inputs
        :return: Deployment URL for invoking the agent
        """
        # Set source URL if provided
        if source:
            self._agent_asset.asset_url = source
        elif not self._agent_asset.asset_url:
            raise mlrun.errors.MLRunInvalidArgumentError(
                "Source archive must be provided either in agent asset or as 'source' parameter"
            )
        # Validate mandatory configurations
        configs = self._validate_mandatory_configs(
            kwargs, self._agent_asset.mandatory_configurations
        )

        # Get or create project
        project_obj = mlrun.get_or_create_project(project)

        # Determine which image to use (transparent optimization)
        if base_image_with_requirements:
            # User provided pre-built image
            logger.info(
                "Using provided base image with requirements",
                agent=self.name,
                image=base_image_with_requirements,
            )
            final_image = base_image_with_requirements
            needs_source = True
        elif self._built_final_image:
            # Already built complete image (requirements + source)
            logger.info(
                "Reusing cached image with requirements and source",
                agent=self.name,
                image=self._built_final_image,
            )
            final_image = self._built_final_image
            needs_source = False
        elif self._built_base_image:
            # Have base image, need to add source
            logger.info(
                "Reusing cached base image, building with source",
                agent=self.name,
                base_image=self._built_base_image,
            )
            final_image = self._build_with_source(project_obj, self._built_base_image)
            needs_source = False
        else:
            # Build from scratch: base + source
            logger.info(
                "Building from scratch (base + source)",
                agent=self.name,
            )
            base_image = self._build_base_image(
                project_obj,
                kwargs.get("base_image"),
                kwargs.get("requirements"),
            )
            final_image = self._build_with_source(project_obj, base_image)
            needs_source = False

        # Set up application function for deployment
        app = project_obj.set_function(
            kind="application",
            image=final_image,
            name=self.name,
        )

        # Add source if using pre-built base image
        if needs_source:
            app.with_source_archive(
                source=self._agent_asset.asset_url, pull_at_runtime=False
            )

        # Configure application port
        app.set_internal_application_port(
            kwargs.get("port") or self._agent_asset.default_port
        )

        # Configure command and args
        app.spec.command = kwargs.get("command") or self._agent_asset.default_command
        app.spec.args = kwargs.get("args") or self._agent_asset.default_args

        # Set environment variables from configs
        for key, value in configs.items():
            app.set_env(key, value)

        # Deploy application
        # If gateway_config provided, don't create default gateway (we'll create custom one)
        create_default_gateway = (
            kwargs.get("create_default_api_gateway", False)
            if not gateway_config
            else False
        )
        app.deploy(with_mlrun=False, create_default_api_gateway=create_default_gateway)

        # Create API gateway if config is provided
        if gateway_config:
            # Set default gateway name if not provided
            gateway_params = gateway_config.copy()
            if "name" not in gateway_params:
                gateway_params["name"] = f"{self.name}-gateway"

            app.create_api_gateway(**gateway_params)

        # Get the deployment URL
        deployment_url = app.status.url if hasattr(app.status, "url") else None
        if not deployment_url and hasattr(app.status, "external_invocation_urls"):
            # Fallback to first external URL
            urls = app.status.external_invocation_urls
            deployment_url = urls[0] if urls else None

        logger.info(
            "Agent deployed successfully",
            agent=self.name,
            project=project,
            url=deployment_url,
        )

        return deployment_url


def import_agent(name: str) -> MarketplaceAgentDeployer:
    """
    Import an agent from the MLRun marketplace.

    :param name: Agent name (e.g., "marketplace://atomic-writer:0.0.1")
    :return: MarketplaceAgentDeployer instance

    Example:
        >>> agent = mlrun.import_agent("marketplace://atomic-writer:0.0.1")
        >>> agent.info()
        >>> agent.deploy(project="my-project", OPENAI_API_KEY="sk-...", ...)
    """
    agent_asset = MarketplaceBackend.get(name)
    return MarketplaceAgentDeployer(agent_asset)


def deploy_agent(
    name: str,
    project: str,
    source: Optional[str] = None,
    gateway_config: Optional[dict[str, Any]] = None,
    **kwargs,
):
    """
    Convenience function to import and deploy an agent in one call.

    This function is designed for simple, one-time deployments. For reusing
    built images across multiple deployments, use import_agent() and call
    deploy() multiple times on the same agent instance.

    :param name: Agent name (e.g., "marketplace://atomic-writer:0.0.1")
    :param project: MLRun project name
    :param source: Source archive URL/path (required if not set in agent asset)
    :param gateway_config: API gateway configuration dict
        (see MarketplaceAgentDeployer.deploy for details)
    :param kwargs: Additional configuration options including:
        - base_image: Override default base image (e.g., "ubuntu:22.04")
        - requirements: Override requirements (list or file path)
        - Any input configurations (secrets/env vars)
        (see MarketplaceAgentDeployer.deploy for full options)
    :return: Deployment URL for invoking the agent

    Example:
        >>> mlrun.deploy_agent(
        ...     "marketplace://atomic-writer:0.0.1",
        ...     project="my-project",
        ...     source="v3io:///projects/my-project/artifacts/agent.tar.gz",
        ...     gateway_config={
        ...         "authentication_mode": "none",
        ...         "path": "/",
        ...         "ssl_redirect": True,
        ...     },
        ...     OPENAI_API_KEY="sk-...",
        ... )
    """
    agent_deployer = import_agent(name)
    return agent_deployer.deploy(
        project=project,
        source=source,
        gateway_config=gateway_config,
        **kwargs,
    )
