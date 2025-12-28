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
(assert (= (list.length.int x) 3))
(assert (> (list.length.int x) 0))
(assert (> (list.length.int x) 1))
(assert (> (list.length.int x) 2))
(assert (not (= (list.get.int x 0) 0)))
(assert (not (= (list.get.int x 0) 0)))
(assert (not (= (list.get.int x 0) 0)))
(assert (and (and (and (= (mod 200004931 (list.get.int x 0)) (mod 66679984 (list.get.int x 0))) (= (mod 66679984 (list.get.int x 0)) 0)) (= (+ (* (list.get.int x 1) 200004931) (* (list.get.int x 2) 66679984)) (list.get.int x 0))) (> (list.get.int x 0) 0)))
(check-sat)
(get-model)