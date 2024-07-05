from lavis.models.blip2_models.blip2 import Blip2Base

class Blip4Fed(Blip2Base):
    def __init__(self, ):
        super.__init__(Blip4Fed, self)
        self.q_former = self.init_Qformer()

    def forward(self, image, text):
        image