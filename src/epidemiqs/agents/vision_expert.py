import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
from typing import List
import json 
import traceback
import openai  # Import openai to check version
from packaging import version  # For version comparison
from epidemiqs.utils.llm_models import choose_model


vision_expert_system_prompt="""You should Analyse the image and provide insights. \n
                        Be precise and accurate in your response.\n
                        if user asked about specific criteria , provide the required information from image such as:"answer to user request in descriptive way"," "metric 1": value of metric 1, "metric 2": value of metric 2, ...} (do not forget to give the unit for values.) \n
                        these metrics should be extracted based on the user request. if the requested metric can not be extracted from the data,\n
                        you should respond with "I can not extract that metric from the user request (along with your reason why you can not do so)"\n
                        you might receive multiple images, in that case analyse each image and provide insights for each one, and also provide comparative analysis of figures,\n
                        then compare between them that how they are evolving. Ensure that you provide accurate values, if the plots show bandwidth or region rather than solid line, or variation, describe those bandwidths (usually represent uncertainty) with numerical details. \n
                        Never hallucinate or make up values, if the plots are not provided or you can not extract the requested metric, you should respond with "I can not extract that metric from the user request (along with your reasoning why you can not)".\n"""
load_dotenv()
def check_file_format(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.png':
        return 'PNG'
    elif ext == '.pdf':
        return 'PDF'
    else:
        return 'Unknown'

class VisionAgent:
    def __init__(self,name="VisionExpert",llm=choose_model()["experts"],system_prompt=vision_expert_system_prompt, role_description="",llm_model= "gpt-4.1-mini"):
        self.client = OpenAI()
        self.name = name
        self.model = llm_model
        self.memory = []
        self.system_prompt = (str(system_prompt+role_description))
        self._update_memory(role="system", content=
                    [{
                    "type": "input_text",  # Corrected from 'text' to 'input_text'
                    "text": f"{self.system_prompt}"
                }]      )


    #def send_message(self, query, image_path=["C:\phd\GEMF_LLM\simulation_output\FastGEMF_plot.png"], image_instructions="This is simualation result showing the output of FastGEMF for ."):
    def send_message(self, query:str="look at plots and provide insight", image_paths:List[str]=[None]):
        print("\n The vision agent is looking at the results ... \n")
        
        content = [{
                    "type": "input_text",  # Corrected from 'text' to 'input_text'
                    "text": query
                }]
     
        
        print(image_paths)  
        
        # all images to the user content
        for path in image_paths:
            if path is not None:
                try:
                    with open(path, "rb") as image_file:
                        file_format = check_file_format(path)
                        
                        if file_format == "PNG":
                            img_b64_str = base64.b64encode(image_file.read()).decode('utf-8')
                            content.append({
                                "type": "input_image",  # It seems that 'input_image' is the correct type for images, I do not not it changes all the time!!
                                "image_url": f"data:image/png;base64,{img_b64_str}",
                                "detail": "high"
                            })
                        
                        elif file_format == "PDF":
                            pdf_data = image_file.read()
                            base64_string = base64.b64encode(pdf_data).decode("utf-8")
                            print(base64_string)  # Debug print of base64 string
                            content.append({
                                "type": "file",  # Corrected from 'input_file' to 'file'
                                "filename": path,
                                "file_data":  f"data:application/pdf;base64,{base64_string}",
                            })
                        
                        else:
                            message = f"Unsupported file format {file_format}. Please provide a PNG or PDF file."
                            print(message)
                            return message
                    
                except FileNotFoundError:
                    message = f"File not found: {path}"
                    print(message)
                    return message
                except Exception as e:
                    error_trace= traceback.format_exc()
                    message = f"Error processing file {path}: {str(error_trace)}"
                    print(message)
                    return message
        

        self._update_memory(role="user", content=content)
        try:
            openai_version = version.parse(openai.__version__)
            if openai_version < version.parse("1.70.0"):
                result = self.client.chat.completions.create(
                    model=self.model,
                    messages=content,
                    stream=False,
                )

                reply = result.choices[0].message.content
            else:
                result = self.client.responses.create(
                    model=self.model,
                    input=self.memory,
                    stream=False,
                    max_output_tokens=200_000
                )

                #response_id = getattr(result, 'id', None)
                #if not response_id or not response_id.startswith("msg_"):
                #    response_id = f"msg_{uuid.uuid4().hex}"
                reply = result.output_text
            self._update_memory(role="assistant", content=[
                            {
                            "type": "output_text",
                            "text": str(reply)
                            }]
                        
                        )
            print("API Response:", reply)  # Debug print of API response
            return reply
            
        except AttributeError as e:
            error_trace= traceback.format_exc()
            error_msg = f"API response error: {str(error_trace)}. Check response structure."
            print(error_msg)
            return error_msg
        except Exception as e:
            error_trace= traceback.format_exc()
            error_msg = f"Error in API call: {str(error_trace)}"
            print(error_msg)
            return error_msg
        
        
            
    def _update_memory(self, role, content):
        
        self.memory.append({"role": role, "content": content})
        
    def not_working_function(self, role, content, msg_id=None,image_paths=None):
        entry = {
            "role": role,
            "content": []
        }
        #if msg_id:
            #entry["id"] = msg_id

        if role == "assistant":
            entry["content"].append({
                "type": "output_text",
                "text": content
            })
        elif role == "user":
            entry["content"].append({
                "type": "input_text",
                "text": content
            })
        elif role == "system":
            entry["content"].append({
                "type": "input_text",
                "text": content
            })
        if image_paths:
            for path in image_paths:
                if path is not None and os.path.exists(path):
                    try:
                        with open(path, "rb") as file:
                            file_format = self.check_file_format(path)

                            if file_format == "PNG":
                                img_b64_str = base64.b64encode(file.read()).decode('utf-8')
                                entry["content"].append({
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{img_b64_str}",
                                    "detail": "high"
                                })

                            elif file_format == "PDF":
                                pdf_b64_str = base64.b64encode(file.read()).decode("utf-8")
                                entry["content"].append({
                                    "type": "file",
                                    "filename": os.path.basename(path),
                                    "file_data": f"data:application/pdf;base64,{pdf_b64_str}",
                                })
                            else:
                                print(f"Unsupported file format: {file_format}")
                    except Exception as e:
                        print(f"Error loading file {path}: {e}")

        self.memory.append(entry)



    def to_json(self,iteration:int=0): #,iteration:int):
        # we already have the iteration as a global variable in the workflow
        path = os.path.join("output", "vision_agent.json") 
        #new_data = json.loads(self.response) 
        new_data = self.response 
        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    existing_data = json.load(f)  
                    if not isinstance(existing_data, dict):
                        existing_data = {}  
                except json.JSONDecodeError:
                    existing_data = {} 
        else:
            existing_data = {}

        existing_data[f"iteration_{iteration}"] = new_data
        with open(path, "w") as f:
            json.dump(existing_data, f, indent=4)


if __name__ == "__main__":
    #with open("C:\phd\GEMF_LLM\simulation_output\image1.png", "rb") as image_file:
        #img_b64 = base64.b64encode(image_file.read()).decode('utf-8')
        #print(img_b64)
        import uuid
        from termcolor import colored
        from os.path import join as ospj
        agent = VisionAgent(role_description="You are a a professional AI image analyser. you are given images and  You analyze the image and provide insights.")
        #print(agent.memory)
        print("\n\n")
        result=agent.send_message("""could two viruses coexist or not ?""",
                            image_paths=[ospj(os.getcwd(),"output","results-11.png"),ospj(os.getcwd(),"output","results-13.png") ] )
        response_id = getattr(result, 'id', f"msg_{uuid.uuid4().hex}")
        print("Response:", result)
        result=agent.send_message("what were the images you processed? what was the model in picture")
        print(colored(result, "green"))
        
        print(f"Response ID: {response_id}")
        final_result=agent.send_message("what about this new image? how is it different from them?",image_paths=[ospj(os.getcwd(),"output","results-12.png") ])
        
        print(colored(f"final result\n: {final_result}","cyan"))# This will use the updated memory
        

