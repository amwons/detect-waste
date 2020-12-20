import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation of metrics', add_help=False)
    parser.add_argument('--addType', default=1, type=int)
    parser.add_argument('--annFile', type=str)
    parser.add_argument('--resFile', type=str)
    
    return parser

def main(args):
    annType = ['segm','bbox','keypoints']
    annType = annType[args.addType]      #specify type here
    print('Running demo for *%s* results.'%(annType))

    #initialize COCO ground truth api
    annFile = args.annFile
    cocoGt=COCO(annFile)

    #initialize COCO detections api
    resFile = args.resFile
    cocoDt=cocoGt.loadRes(resFile)

    imgIds=sorted(cocoGt.getImgIds())

    # running evaluation
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
