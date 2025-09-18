from setuptools import setup,find_packages

def get_packages(file_path):
    '''This function will return the list of requirements'''
    requirements=[]
    
    E_AND_DOT='-e .'
    with open(file_path) as req_obj:
        requirements=req_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        
        if E_AND_DOT in requirements:
            requirements.remove(E_AND_DOT)
    return requirements        
        


setup(
    name="End to end Machine Learning project",
    version="0.0.1",
    author="Ayaan Shaikh",
    author_email="ayaanskh23@gmail.com",
    packages=find_packages(),
    install_requires=get_packages('requirements.txt')
    
)