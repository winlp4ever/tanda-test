select question.id as qid,
    question.question_text as question,
    answer_temp.id as aid,
    answer_temp.answer_text as answer
from question_answer_temp
inner join question 
on question.id = question_answer_temp.question_id
inner join answer_temp 
on answer_temp.id = question_answer_temp.answer_temp_id
where source_type not like '%google%'
order by qid 
limit 1000;