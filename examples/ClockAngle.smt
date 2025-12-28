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
(assert (= (list.length.int x) 2))
(assert (> (list.length.int x) 0))
(assert (> (list.length.int x) 1))
(assert (and (and (< 0 (list.get.int x 0)) (<= (list.get.int x 0) 12)) (and (<= 0 (list.get.int x 1)) (< (list.get.int x 1) 60))))
(assert (or (= (ite (>= (- (+ (* 30 (list.get.int x 0)) (/ (list.get.int x 1) 2)) (* 6 (list.get.int x 1))) 0) (- (+ (* 30 (list.get.int x 0)) (/ (list.get.int x 1) 2)) (* 6 (list.get.int x 1))) (- (- (+ (* 30 (list.get.int x 0)) (/ (list.get.int x 1) 2)) (* 6 (list.get.int x 1))))) 45) (= (ite (>= (- (+ (* 30 (list.get.int x 0)) (/ (list.get.int x 1) 2)) (* 6 (list.get.int x 1))) 0) (- (+ (* 30 (list.get.int x 0)) (/ (list.get.int x 1) 2)) (* 6 (list.get.int x 1))) (- (- (+ (* 30 (list.get.int x 0)) (/ (list.get.int x 1) 2)) (* 6 (list.get.int x 1))))) 315)))
(check-sat)
(get-model)