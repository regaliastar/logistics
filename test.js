
function sort_7000(){
	const length = 700000
	let arr = []
	for(let i = 0; i < length; i++){
		let v = Math.random()
		arr.push(v)
	}
	arr.sort()
	return 0
}

const begin = new Date()
sort_7000()
const end = new Date()
console.log(end - begin,'ms')