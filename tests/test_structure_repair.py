import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import io
import cv2
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app
from background_repair import repair_texture, validate_inputs


class StructureRepairTests(unittest.TestCase):
    def setup_scene(self):
        rng = np.random.default_rng(42)
        # Carpet stripes with irregular fine grain, a different floor underneath.
        arr=np.full((180,240,3),[120,80,40],np.uint8)
        yy,xx=np.mgrid[:140,:240]
        light=180+(yy%6)*4+rng.integers(-4,5,(140,240))
        arr[:140]=np.stack([light,light-18,light-35],axis=2).astype(np.uint8)
        arr[50:120,70:150]=[225,225,225]
        image=Image.fromarray(arr);raw=np.zeros((180,240),np.uint8);raw[50:120,70:150]=255
        box=[10,20,58,68];return image,Image.fromarray(raw),image.crop(box),box

    def test_reference_texture_preserves_boundary_and_outside_pixels(self):
        image,mask,ref,box=self.setup_scene();raw=validate_inputs(image,mask,ref,box)
        result=np.asarray(repair_texture(image,raw,ref));original=np.asarray(image)
        self.assertTrue(np.array_equal(result[raw==0],original[raw==0]))
        self.assertGreater(result[60:110,80:140,0].std(),4)
        self.assertGreater(np.mean(result[60:110,80:140,0]-result[60:110,80:140,2]),25)
        self.assertTrue(np.array_equal(result,np.asarray(repair_texture(image,raw,ref))))

    def test_rejects_overlap_empty_mask_bad_dimensions_and_overlarge_mask(self):
        image,mask,ref,box=self.setup_scene()
        with self.assertRaises(ValueError):validate_inputs(image,mask,image.crop((80,60,120,100)),[80,60,120,100])
        for bad in [Image.new('L',image.size),Image.new('L',image.size,255),Image.new('L',(10,10),255)]:
            with self.assertRaises(ValueError):validate_inputs(image,bad,ref,box)

    def test_endpoint_preview_cache_and_gpu_not_configured(self):
        image,mask,ref,box=self.setup_scene()
        def png(im):
            b=io.BytesIO();im.save(b,format='PNG');return b.getvalue()
        files={k:(k+'.png',png(im),'image/png') for k,im in [('file',image),('mask',mask),('reference',ref)]}
        client=TestClient(app.app)
        data={'reference_box':','.join(map(str,box)),'description':'延续编织地毯','engine':'reference'}
        with patch.object(app,'result_cache',app.MemoryCache(1_000_000,4)):
            result=client.post('/api/structure-repair',files=files,data=data)
            self.assertEqual(result.status_code,200,result.text)
            self.assertFalse(result.json()['cache']['result'])
            again=client.post('/api/structure-repair',files=files,data=data)
            self.assertTrue(again.json()['cache']['result'])
            self.assertEqual(result.json()['clean'],again.json()['clean'])
        with patch.dict(app.os.environ,{'BACKGROUND_WORKER_URL':'','BACKGROUND_WORKER_TOKEN':''}):
            self.assertFalse(client.get('/api/background-capabilities').json()['gpu_generation'])
            data['engine']='gpu';r=client.post('/api/structure-repair',files=files,data=data)
            self.assertEqual(r.status_code,503)
        data['reference_box']='nan,1,2,3';r=client.post('/api/structure-repair',files=files,data=data)
        self.assertEqual(r.status_code,422)

    def test_worker_failure_keeps_input_and_releases_lock(self):
        import httpx
        image,mask,ref,box=self.setup_scene();before=image.tobytes()
        with patch.dict(app.os.environ,{'BACKGROUND_WORKER_URL':'http://worker','BACKGROUND_WORKER_TOKEN':'secret'}), \
             patch.object(app,'detection_prompt',return_value='carpet'), \
             patch('httpx.post',side_effect=httpx.ConnectError('internal server detail')):
            with self.assertRaises(app.HTTPException) as caught:
                app.prepare_structure_repair(image,mask,ref,box,'地毯','gpu')
        self.assertEqual(caught.exception.status_code,502)
        self.assertNotIn('internal server detail',caught.exception.detail)
        self.assertEqual(image.tobytes(),before)
        self.assertFalse(app.model_lock.locked())

    def test_gpu_result_cannot_change_pixels_outside_painted_area(self):
        image,mask,ref,box=self.setup_scene()
        generated=io.BytesIO();Image.new('RGB',image.size,'red').save(generated,format='PNG')
        import httpx,base64
        with patch.dict(app.os.environ,{'BACKGROUND_WORKER_URL':'http://worker','BACKGROUND_WORKER_TOKEN':'test-secret'}), \
             patch.object(app,'detection_prompt',return_value='woven carpet'), \
             patch('httpx.post',return_value=httpx.Response(200,content=generated.getvalue())), \
             patch.object(app,'result_cache',app.MemoryCache(1_000_000,4)):
            payload=app.prepare_structure_repair(image,mask,ref,box,'地毯','gpu')
        output=np.asarray(Image.open(io.BytesIO(base64.b64decode(payload['clean'].split(',')[1]))))
        outside=np.asarray(mask)==0
        self.assertTrue(np.array_equal(output[outside],np.asarray(image)[outside]))
        self.assertFalse(app.model_lock.locked())

if __name__=='__main__':unittest.main()
