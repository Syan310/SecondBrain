from fastapi import APIRouter

router = APIRouter(prefix="/quiz", tags=["quiz"])


@router.post("/start")
async def start_quiz():
    """开始一个Quiz Session"""
    return {"message": "quiz start endpoint placeholder"}


@router.post("/answer")
async def submit_answer():
    """提交答案并获取评判结果"""
    return {"message": "quiz answer endpoint placeholder"}


@router.post("/end")
async def end_quiz():
    """结束Quiz Session, 返回复习报告"""
    return {"message": "quiz end endpoint placeholder"}
