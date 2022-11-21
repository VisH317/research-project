from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import gym
import gym.spaces as spaces
import numpy as np

from Generator import Generator, GeneratorPreprocess

kitchens = [f"FloorPlan{i}" for i in range(1, 21)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(1, 21)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(1, 21)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(1, 21)]

trainMats = kitchens + living_rooms + bedrooms + bathrooms

kitchens = [f"FloorPlan{i}" for i in range(21, 26)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(21, 26)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(21, 26)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(21, 26)]

valMats = kitchens + living_rooms + bedrooms + bathrooms

kitchens = [f"FloorPlan{i}" for i in range(26, 31)]
living_rooms = [f"FloorPlan{200 + i}" for i in range(26, 31)]
bedrooms = [f"FloorPlan{300 + i}" for i in range(26, 31)]
bathrooms = [f"FloorPlan{400 + i}" for i in range(26, 31)]

testMats = kitchens + living_rooms + bedrooms + bathrooms

class AI2thor(gym.Env):
    def __init__(self, ttv):
        super(AI2thor, self).__init__()
        
        if ttv=='train':
            self.rooms = trainMats
        elif ttv=='val':
            self.rooms = valMats
        else:
            self.rooms = testMats
            
        self.actions = ["MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateLeft", "RotateRight", "LookUp", "LookDown", "Crouch", "Stand", "Done"]
        self.action_space = spaces.Discrete(11)
        
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(3, 448, 448), dtype=np.uint8),
            "text": spaces.Discrete(2048)
        })
        
        self.text_processor = GeneratorPreprocess()
            
        self.ttv = ttv
        self.controller = None
        self.current_obj = None
        self.obj_position = None
        self.done = False
        self.obj_vect = None
                
    def step(self, action):
        event = self.controller.step(self.actions(action))
        reward = -1
        done = False
        if action=="Done":
                ag_pos = event.metadata["agent"]["position"]
                if self.compute_distance(ag_pos):
                    reward = 15
                done = True
        info = None
        obs = {
            "image": event.frame,
            "text": self.obj_vect
        }
        
        return obs, reward, done, info
        

    def reset(self):
        self.controller = None
        self.current_obj = None
        self.done = False
        # check reward if within the target distance, calculate target distance relative to size
        self.choose_scene()
        event, obj_vect = self.domain_rand()
        self.obj_vect = obj_vect
        obs = {
            "image": event.frame,
            "text": self.obj_vect
        }
        return obs
        
    def choose_scene(self):
        roomID = self.rooms[np.random.randint(0, len(self.rooms))]
        self.controller = Controller(
            scene=roomID,
            rotateStepDegrees=60,
            width=448,
            height=448,
            platform=CloudRendering
        )
        
    def compute_distance(self, ag_pos):
        x = abs(ag_pos['x']-self.obj_position['x'])
        z = abs(ag_pos['z']-self.obj_position['z'])
        return np.sqrt(x**2+z**2)<=1
        
    def domain_rand(self):
        if self.ttv=='train':
            event = self.controller.step(
                action="RandomizeMaterials",
                useTrainMaterials=True,
                useValMaterials=False,
                useTestMaterials=False
            )
            obj_vect = self.choose_object(event)
            return event, obj_vect
        elif self.ttv=='val':
            event = self.controller.step(
                action="RandomizeMaterials",
                useTrainMaterials=False,
                useValMaterials=True,
                useTestMaterials=False
            )
            obj_vect = self.choose_object(event)
            return event, obj_vect
        elif self.ttv=='test':
            event = self.controller.step(
                action="RandomizeMaterials",
                useTrainMaterials=False,
                useValMaterials=False,
                useTestMaterials=True
            )
            obj_vect = self.choose_object(event)
            return event, obj_vect
        
    def choose_object(self, event):
        total_objects = len(event.metadata["objects"])
        ix = np.random.randint(0, total_objects)
        self.current_obj = event.metadata["objects"][ix]
        obj_vect = self.text_processor(event.metadata["objects"][ix]["objectType"])
        return obj_vect
        