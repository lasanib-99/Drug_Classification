from setuptools import find_packages, setup # type: ignore
from typing import List

def get_requirements(file_path:str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    HYPHEN_E_DOT = "-e ."
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements] ## Lines are read without unwanted space (new line in this case)
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)  ## The '-e .' part should be ignored
            return requirements
        
setup(
    name = 'Diabetes_Existence',
    version = '0.0.1',
    author = 'Lasani',
    author_email = 'lasani@mihcm.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)