(set-logic ALL)

(declare-datatypes ((List 1))
    ((par (T) ((cons (head T) (tail (List T))) (nil)))))


(define-fun-rec list.length.int ((l (List Int))) Int
  (ite (= l (as nil (List Int)))
       0
       (+ 1 (list.length.int (tail l)))))


(define-fun-rec list.get.int ((l (List Int)) (idx Int)) Int
  (ite (< idx 0)
       (list.get.int l (+ (list.length.int l) idx))
  (ite (= idx 0)
       (head l)
       (list.get.int (tail l) (- idx 1)))))

(declare-const x (List Int))
(assert (> (list.length.int x) 0))
(assert (= (list.get.int x 0) -15))
(assert (> (list.length.int x) 1))
(assert (> (list.length.int x) (- 1 1)))
(assert (=> (> -6 (list.get.int x (- 1 1))) (= (list.get.int x 1) -6)))
(assert (=> (> -6 (list.get.int x (- 1 1))) true))
(assert (> (list.length.int x) 0))
(assert (= (list.get.int x 0) -15))
(assert (> (list.length.int x) 1))
(assert (> (list.length.int x) (- 1 1)))
(assert (=> (not (> -6 (list.get.int x (- 1 1)))) (= (list.get.int x 1) (list.get.int x (- 1 1)))))
(assert (=> (not (> -6 (list.get.int x (- 1 1)))) true))
(check-sat)
(get-model)