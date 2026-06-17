from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    requirement_list = []

    try:
        with open("requirements.txt", "r") as file:
            requirements = file.readlines()

            for requirement in requirements:
                requirement = requirement.strip()
                if requirement and requirement != "-e .":
                    requirement_list.append(requirement)

    except FileNotFoundError:
        print("requirements.txt file not found.")

    return requirement_list


print(get_requirements())

setup(
    name="AI-TRAVEL-PLANNER",
    version="0.0.1",
    author="nikita Parmar",
    author_email="nikitaparmar23456@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)